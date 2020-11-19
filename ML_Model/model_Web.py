"""Vinculación con flask"""
import flask
import numpy as np
import pandas as pd
import joblib

params = joblib.load('params_dt.joblib')
app= flask.Flask(__name__)

def categorize(df):
    df['Sex'] = df["Sex"].replace(['male','female'], [1,0])
    df['Saving accounts'] = df["Saving accounts"].replace(['n/a','little','moderate','quite rich','rich'], [0,1,2,3,4])
    df['Checking account'] = df["Checking account"].replace(['n/a','little','moderate','rich'], [0,1,2,3])
    df['Housing'] = df["Housing"].replace(["free","rent","own"],[0,1,2])
    df['Purpose'] = (df["Purpose"].replace(["car","radio/TV","furniture/equipment","business","education","repairs",
                                      "domestic appliances", "vacation/others"],[0,1,2,3,4,5,6,7]))
    return(df)

def z_score(feature,mean,std):
    new_feature = (feature - mean) / std
    return new_feature

@app.route('/<data>',methods=['POST'])
def Home(data):
    cont=flask.request.json
    df = pd.DataFrame()
    df = df.append(cont, ignore_index=True)
    df = categorize(df)
    for column in (df.columns):
        mean=params['means'][column]
        std=params['stds'][column]
        df[column]= z_score(df[column],mean,std)
    df = df.drop(['Job'],axis = 1)
    df = df = df[['Age', 'Sex', 'Housing', 'Saving accounts', 'Checking account', 'Duration', 'Purpose']]
    test = df.to_numpy()
    model = joblib.load('model_dt.joblib')
    ans = (model.predict(test.reshape(1,-1)))[0]
    int_ans= int(ans)
    return flask.jsonify({'Prediction':int_ans})

