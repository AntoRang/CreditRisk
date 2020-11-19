
"""
    The code shows the backend server code
    The development requires Flask
    The ML and data paramenters (mean/std) must be imported 
    Inputs: "Sex", "Saving accounts", "Checking account", "Housing", "Purpose","Duration", "Age"
    Output: "Credit ammount" - Max money allowed and risk type    
"""

#Libraries
import flask
import numpy as np
import pandas as pd
import joblib

#Parameters from the data (mean/std)
params = joblib.load('params_dt.joblib')

#Variable name for Flask
app= flask.Flask(__name__)

#Function for categorizing to numerical data
def categorize(df):
    df['Sex'] = df["Sex"].replace(['male','female'], [1,0])
    df['Saving accounts'] = df["Saving accounts"].replace(['n/a','little','moderate','quite rich','rich'], [0,1,2,3,4])
    df['Checking account'] = df["Checking account"].replace(['n/a','little','moderate','rich'], [0,1,2,3])
    df['Housing'] = df["Housing"].replace(["free","rent","own"],[0,1,2])
    df['Purpose'] = (df["Purpose"].replace(["car","radio/TV","furniture/equipment","business","education","repairs",
                                      "domestic appliances", "vacation/others"],[0,1,2,3,4,5,6,7]))
    return(df)

#Function standarizing data with "z-score"
def z_score(feature,mean,std):
    new_feature = (feature - mean) / std
    return new_feature

#Generation of the main page
@app.route('/<data>',methods=['POST'])
def Home(data):
    #Data adquisition
    cont=flask.request.json
    #Converting into dataframe and treating data as it
    df = pd.DataFrame()
    df = df.append(cont, ignore_index=True)
    #Categorizing data
    df = categorize(df)
    #Standarizing data
    for column in (df.columns):
        mean=params['means'][column]
        std=params['stds'][column]
        df[column]= z_score(df[column],mean,std)
    #Dropping colums no requirind for the ML model
    df = df.drop(['Job'],axis = 1)
    #Indexing the order properly
    df = df = df[['Age', 'Sex', 'Housing', 'Saving accounts', 'Checking account', 'Duration', 'Purpose']]
    #Converting to a numpy array
    test = df.to_numpy()
    #Predicting the output
    model = joblib.load('model_dt.joblib')
    pred = (model.predict(test.reshape(1,-1)))[0]
    int_pred = int(pred)
    #Generating the final output
    risk_type = params["type"][int_pred] 
    min_credit = params['min'][risk_type]
    max_credit = params['max'][risk_type]
    #Returning the result
    return flask.jsonify({'Risk':risk_type, 'Min. ammount': min_credit, 'Max. ammount': max_credit})

