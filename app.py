from flask import Flask, request, jsonify
import pandas as pd
import statsmodels.api as sm
app = Flask(__name__)


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
    model = sm.MixedLM.from_formula(
        'values ~ 1', df, groups=df['clusters']
    )
    res = model.fit(reml=method)
    tau = res.cov_re.groups[0]
    sigma2 = res.scale
    return jsonify([tau, sigma2])


if __name__ == "__main__":
    app.run()
