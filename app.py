# TODO: Do cross-levels
# TODO: Return Convergence information

import re
from flask import Flask, request, jsonify
from statsmodels.formula.api import ols
from scipy.stats import f
import numpy as np
import pandas as pd
import statsmodels.api as sm

app = Flask(__name__)
ALPHA = 0.05


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/icc", methods=['POST'])
def icc():
    req = request.get_json()
    method = req['method']
    clusters = pd.Series(req['x'])
    values = pd.Series(req['y'])
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
    result = {
        'ICC': icc, 'LowerCI': low_ci, 'UpperCI': up_ci, 'N': a, 'k': k,
        'vara': var_a, 'varw': ms_w
    }
    # print result
    if method == 'ANOVA':
        return jsonify(result)
    model = sm.MixedLM.from_formula(
        'values ~ 1', df, groups=df['clusters']
    )
    res = model.fit(reml=method)
    tau = res.cov_re.groups[0]
    sigma2 = res.scale
    result['vara'] = tau
    result['varw'] = sigma2
    result['ICC'] = tau / (tau + sigma2)
    return jsonify(result)


@app.route("/r2", methods=['POST'])
def r2():
    req = request.get_json()
    data = req['data']
    headers = list(map((lambda x: del_utf(x.encode('utf-8'))), req['headers']))
    data = pd.DataFrame(data, columns=headers)
    cluster_var = del_utf(req['cluster_var'].encode('utf-8'))
    outcome_var = del_utf(req['outcome_var'].encode('utf-8'))
    null_equation = del_utf(req['null_equation'].encode('utf-8'))
    a = float(data[cluster_var].nunique())
    temp = data.groupby(cluster_var).count()
    vals = temp[outcome_var]
    vals2 = vals.apply(square)
    k = (1 / (a - 1)) * (vals.sum() - (vals2.sum() / vals.sum()))
    model_b = sm.MixedLM.from_formula(
        null_equation, data, groups=data[cluster_var]
    )
    res_b = model_b.fit(reml=False)
    int_preds = req['int_preds']
    l_one_preds = req['l_one_preds']
    eqn_data = create_fit_equation(
        int_preds, l_one_preds, cluster_var, outcome_var, data)
    fit_eqn = eqn_data[0]
    data = eqn_data[1]
    # print data
    model_f = sm.MixedLM.from_formula(
        fit_eqn, data, groups=data[cluster_var]
    )
    res_f = model_f.fit(reml=False)
    print(res_f.summary())
    print(res_f.converged)
    tau_b = res_b.cov_re.groups[0]
    sigma2_b = res_b.scale
    tau_f = res_f.cov_re.groups[0]
    sigma2_f = res_f.scale
    result = {}
    level_one_r_2 = 1 - ((tau_f+sigma2_f)/(tau_b+sigma2_b))
    level_two_r_2 = 1 - ((tau_f+(sigma2_f/k))/(tau_b+(sigma2_b/k)))
    result['a'] = a
    result['k'] = k
    result['vara_b'] = tau_b
    result['varw_b'] = sigma2_b
    result['vara_f'] = tau_f
    result['varw_f'] = sigma2_f
    result['level_one_r_2'] = level_one_r_2
    result['level_two_r_2'] = level_two_r_2
    return jsonify(result)


def square(x):
    return x**2


def del_utf(arg):
    return re.sub(r'[^\x00-\x7F]+', '', arg)


def create_fit_equation(int_preds, l_one_preds, c_var, o_var, data):
    CNT = '_iOxZAnf2FqHrXuxmnXY85Od4hp45C5IoxfptIb10Wj0_'
    for i, value in enumerate(int_preds[0]):
        int_preds[0][i] = del_utf(value.encode('utf-8'))
    for key in l_one_preds.keys():
        temp = l_one_preds[key]
        del l_one_preds[key]
        l_one_preds[del_utf(key.encode('utf-8'))] = temp
    for key in l_one_preds:
        l_one_preds[key][0][0] = list(map(
            (lambda x: del_utf(x.encode('utf-8'))), l_one_preds[key][0][0]))
    int_preds = np.transpose(int_preds)
    for key in l_one_preds:
        l_one_preds[key][0] = np.transpose(l_one_preds[key][0])
    int_eqn = []
    for value in int_preds:
        if value[1] == '2':
            data[value[0] + CNT + '2'] = data[value[0]] - data[value[0]].mean()
            int_eqn.append(value[0] + CNT + '2')
        else:
            int_eqn.append(value[0])
    l_one_eqn = []
    # crosses = []
    for key in l_one_preds:
        if l_one_preds[key][1] == 2:
            new_key = key + CNT + '2'
            l_one_eqn.append(new_key)
            data[new_key] = data[key] - data[key].mean()
        elif l_one_preds[key][1] == 1:
            new_key = key + CNT + '1'
            l_one_eqn.append(new_key)
            data[new_key] = data[key]
            c_means = data.groupby(c_var)[key].mean()
            data[new_key] = data.apply(lambda x: x[key] - c_means[x[c_var]],
                                       axis=1)
        else:
            l_one_eqn.append(key)
    print data.head(n=5)
    l_one_eqn = ' + '.join(map(str, l_one_eqn))
    int_eqn = ' + '.join(map(str, int_eqn))
    predictors = [l_one_eqn, int_eqn]
    return [o_var + ' ~ ' + ' + '.join(map(str, predictors)), data]


if __name__ == "__main__":
    app.run()
