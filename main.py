# TODO: Allow optimization options for ICC or return ICC in Second part, since
# CI for ICC only calculated with ANOVA anyway.

import logging
import requests
from flask import Flask, request
from statsmodels.formula.api import ols
from scipy.stats import f
from scipy.stats import hmean
import numpy as np
import pandas as pd
import statsmodels.api as sm
import math

app = Flask(__name__)
ALPHA = 0.05
CNT = '_efc_centered_'


@app.route("/")
def hello():
    return "Hi:<br>This application is mostly useful if you're accessing "\
        "it from <a href='https://effect-size-calculator.herokuapp.com'>"\
        'effect-size-calculators.herokuapp.com</a>. This repo is the HLM '\
        "workhorse for that application.<br>Thank you, James Uanhoro."


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500


@app.route("/icc", methods=['POST'])
def icc():
    req = request.get_json()
    method = req['method']
    clusters = pd.Series(req['x'])
    values = pd.Series(req['y'])
    channel = req['channel']
    url = req['url']
    icc_background(clusters, method, values, channel, url)
    return 'Move along.', 302


def icc_background(clusters, method, values, channel, url):
    d = {'clusters': clusters, 'values': values}
    df = pd.DataFrame(d)
    lm = ols('values ~ clusters', data=df).fit()
    table = sm.stats.anova_lm(lm, typ=3)
    ss_a = table.sum_sq[1]
    ss_w = table.sum_sq[2]
    num_df = table.df[1]
    denum_df = table.df[2]
    ms_a = ss_a / num_df
    ms_w = ss_w / denum_df
    temp = df.groupby('clusters').count()
    a = float(df['clusters'].nunique())
    vals = temp['values']
    vals2 = vals.apply(square)
    k = (1 / (a - 1)) * (vals.sum() - (vals2.sum() / vals.sum()))
    var_a = (ms_a - ms_w) / k
    icc = var_a / (var_a + ms_w)
    low_F = f.ppf(1 - ALPHA / 2, num_df, denum_df)
    N = len(df.index)
    nbar = N / a
    n_not = nbar - ((vals.subtract(nbar)).apply(square) / ((a - 1) * N)).sum()
    # up_F = f.ppf(1 - ALPHA / 2, denum_df, num_df)
    up_F_2 = f.ppf(ALPHA / 2, num_df, denum_df)
    f_l = (ms_a / ms_w) / low_F
    f_u = (ms_a / ms_w) / up_F_2
    low_ci = (f_l - 1) / (f_l + n_not - 1)
    up_ci = (f_u - 1) / (f_u + n_not - 1)
    deff = icc * (k - 1) + 1
    deft = math.sqrt(deff)
    result = {
        'icc_est': icc, 'lower': low_ci, 'upper': up_ci, 'n': a, 'k': k,
        'vara': var_a, 'varw': ms_w, 'deft': deft, 'des_eff': deff
    }
    headers = {'Content-Type': 'application/json'}
    if method == 'ANOVA':
        requests.post(url + '/faye', json={
            'channel': '/' + channel,
            'data': {
                'tasks_to_do': 1, 'tasks_done': 1, 'result': result
            }
        }, headers=headers)
        return
    model = sm.MixedLM.from_formula(
        'values ~ 1', df, groups=df['clusters']
    )
    requests.post(url + '/faye', json={
        'channel': '/' + channel,
        'data': {
            'tasks_to_do': 1, 'tasks_done': 2
        }
    }, headers=headers)
    res = model.fit(reml=method, method='nm')
    tau = res.cov_re.groups[0]
    sigma2 = res.scale
    result['vara'] = tau
    result['varw'] = sigma2
    result['icc_est'] = tau / (tau + sigma2)
    deff = tau / (tau + sigma2) * (k - 1) + 1
    deft = math.sqrt(deff)
    result['des_eff'] = deff
    result['deft'] = deft
    requests.post(url + '/faye', json={
        'channel': '/' + channel,
        'data': {
            'tasks_to_do': 2, 'tasks_done': 2, 'result': result
        }
    }, headers=headers)
    return


@app.route("/r2", methods=['POST'])
def r2():
    req = request.get_json()
    data = req['data']
    headers = list(map((lambda x: str(x)), req['headers']))
    data = pd.DataFrame(data, columns=headers)
    cluster_var = str(req['cluster_var'])
    outcome_var = str(req['outcome_var'])
    null_equation = str(req['null_equation'])
    optim = req['optim']
    int_preds = req['int_preds']
    l_one_preds = req['l_one_preds']
    channel = req['channel']
    url = req['url']
    r2_background(cluster_var, outcome_var, null_equation, optim, int_preds,
                  l_one_preds, headers, data, channel, url)
    return 'Move along.', 302


def r2_background(cluster_var, outcome_var, null_equation, optim, int_preds,
                  l_one_preds, headers, data, channel, url):
    a = float(data[cluster_var].nunique())
    k = np.average(hmean(data.groupby(cluster_var).count()))
    print(cluster_var, outcome_var, headers)
    print(data.head(5))
    model_b = sm.MixedLM.from_formula(
        null_equation, data, groups=data[cluster_var]
    )
    headers = {'Content-Type': 'application/json'}
    requests.post(url + '/faye', json={
        'channel': '/' + channel,
        'data': {
            'tasks_to_do': 1, 'tasks_done': 2
        }
    }, headers=headers)
    optimizers = ['nm', 'powell', 'cg', 'bfgs']
    res_b = model_b.fit(reml=False, method=optimizers[optim])
    eqn_data = create_fit_equation(
        int_preds, l_one_preds, cluster_var, outcome_var, data)
    fit_eqn = eqn_data[0]
    data = eqn_data[1]
    model_f = sm.MixedLM.from_formula(
        fit_eqn, data, groups=data[cluster_var]
    )
    res_f = model_f.fit(reml=False, method=optimizers[optim])
    tau_b = res_b.cov_re.groups[0]
    sigma2_b = res_b.scale
    tau_f = res_f.cov_re.groups[0]
    sigma2_f = res_f.scale
    result = {}
    level_one_r_2 = 1 - ((tau_f+sigma2_f)/(tau_b+sigma2_b))
    level_two_r_2 = 1 - ((tau_f+(sigma2_f/k))/(tau_b+(sigma2_b/k)))
    result['n'] = a
    result['k'] = k
    result['vara_b'] = tau_b
    result['varw_b'] = sigma2_b
    result['vara_f'] = tau_f
    result['varw_f'] = sigma2_f
    result['level_one_r_2'] = level_one_r_2
    result['level_two_r_2'] = level_two_r_2
    result['convergence_b'] = res_b.converged
    result['convergence_f'] = res_f.converged
    result['icc_b'] = tau_b / (tau_b + sigma2_b)
    result['icc_f'] = tau_f / (tau_f + sigma2_f)
    try:
        model_mat = np.matrix(res_f.model.exog)
        fe = np.matrix(res_f.fe_params)
        sf = np.var(model_mat * fe.getT())
        z = model_mat[:, 0]
        sl = np.sum(np.sum(np.diag(z * tau_f * z.getT()))/model_mat.shape[0])
        sd = 0
        total_var = sf + sl + sigma2_f + sd
        rsq_marg = sf / total_var
        rsq_cond = (sf + sl) / total_var
        result['rsq_marg'] = rsq_marg
        result['rsq_cond'] = rsq_cond
    except Exception:
        pass
    base_results = "Base model:\n" + str(res_b.summary())
    fitted_results = "\nFitted model:\n" + str(res_f.summary())
    cent_0 = "\nA note about modified variable names\n"
    cent_1 = CNT + "1 after a variable name signifies group-mean centering;\n"
    cent_2 = CNT + '2 after a variable name signifies grand-mean centering.'
    cent = cent_0 + cent_1 + cent_2
    result['results'] = base_results + fitted_results + cent
    requests.post(url + '/faye', json={
        'channel': '/' + channel,
        'data': {
            'tasks_to_do': 2, 'tasks_done': 2, 'result': result
        }
    }, headers=headers)
    return


def square(x):
    return x**2


def create_fit_equation(int_preds, l_one_preds, c_var, o_var, data):
    for i, value in enumerate(int_preds[0]):
        int_preds[0][i] = str(value)
    for key in l_one_preds.keys():
        temp = l_one_preds[key]
        del l_one_preds[key]
        l_one_preds[str(key)] = temp
    int_preds = np.transpose(int_preds)
    int_eqn = []
    for value in int_preds:
        if value[1] == '2':
            data[value[0] + CNT + '2'] = data[value[0]] - data[value[0]].mean()
            int_eqn.append(value[0] + CNT + '2')
        else:
            int_eqn.append(value[0])
    l_one_eqn = []
    crosses = []
    for key in l_one_preds:
        # Ensure text is ASCII
        l_one_preds[key][0][0] = list(map(
            (lambda x: str(x)), l_one_preds[key][0][0]))
        # Transposing makes life easy
        l_one_preds[key][0] = np.transpose(l_one_preds[key][0])
        l_two_preds = l_one_preds[key][0]
        if l_one_preds[key][1] == 2:
            new_key = key + CNT + '2'
            l_one_eqn.append(new_key)
            data[new_key] = data[key] - data[key].mean()
            results = cross_ints(new_key, l_two_preds, crosses, data)
            crosses = results[0]
            data = results[1]
        elif l_one_preds[key][1] == 1:
            new_key = key + CNT + '1'
            l_one_eqn.append(new_key)
            data[new_key] = data[key]
            c_means = data.groupby(c_var)[key].mean()
            data[new_key] = data.apply(lambda x: x[key] - c_means[x[c_var]],
                                       axis=1)
            results = cross_ints(new_key, l_two_preds, crosses, data)
            crosses = results[0]
            data = results[1]
        else:
            l_one_eqn.append(key)
            results = cross_ints(key, l_two_preds, crosses, data)
            crosses = results[0]
            data = results[1]
    l_one_eqn = ' + '.join(map(str, l_one_eqn))
    int_eqn = ' + '.join(map(str, int_eqn))
    crosses = ' + '.join(map(str, crosses))
    predictors = []
    if len(l_one_eqn) > 0:
        predictors.append(l_one_eqn)
    if len(int_eqn) > 0:
        predictors.append(int_eqn)
    if len(crosses) > 0:
        predictors.append(crosses)
    return [o_var + ' ~ ' + ' + '.join(map(str, predictors)), data]


def cross_ints(val_a, l_two_preds, result, data):
    # Cross-level interactions + insert into data
    for value in l_two_preds:
        if value[1] == '0':
            result.append(val_a + ' : ' + value[0])
        else:
            new_val = value[0] + CNT + '2'
            if new_val not in data:
                data[new_val] = data[value[0]] - data[value[0]].mean()
            result.append(val_a + ' : ' + new_val)
    return([result, data])


if __name__ == "__main__":
    app.run()
