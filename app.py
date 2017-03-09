from flask import Flask, request, jsonify
from statsmodels.formula.api import ols
from scipy.stats import f
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


def square(x):
    return x**2


if __name__ == "__main__":
    app.run()
