"""Vinculación con flask"""
import flask
import numpy as np
import joblib

app= flask.Flask(__name__)

@app.route('/<dato>',methods=['POST'])
def Home(dato):
    cont=flask.request.json
    print(cont)
    test = np.array([2.67716444, 0.66994484, 1.67683077, -0.13364358, -0.70614255,  1.8910083, 0.75438593, -1.45490372])
    model = joblib.load('model_dt.joblib')
    ans = (model.predict(test.reshape(1,-1)))[0]
    int_ans= int(ans)
    print(int_ans)
    return flask.jsonify({'Prediction':int_ans})