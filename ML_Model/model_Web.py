"""Vinculación con flask"""
import flask
import numpy as np
import joblib

app= flask.Flask(__name__)

@app.route('/<data>',methods=['POST'])
def Home(data):
    cont=flask.request.json
    print(cont)
    test = np.array([-0.39963189,0.66994484,0.58531017,0.83775605,1.04384958,-1.23585947,1.02436272])
    model = joblib.load('model_dt.joblib')
    ans = (model.predict(test.reshape(1,-1)))[0]
    int_ans= int(ans)
    print(int_ans)
    return flask.jsonify({'Prediction':int_ans})