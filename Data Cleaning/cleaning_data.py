"""
    This code help us to clean the data regarding on the dataset of CreditRisk.
    The original dataset could be download for free from Kaggle: https://www.kaggle.com/nitinkhandagale/credit-risk-with-label
    The original dataset contains the following paramenters: 
        -Qulitative
            Nominal: 
                Sex: Male / Female
            Ordinal:
                Housing: Free/ Own / Rent
                Saving accounts: Little/ Moderate / Quite Rich / Rich
                Checking accounts: Little / Moderate / Rich
                Purpose
                Risk: Bad/Good
        -Quantitative
            Numerical:
                Age
                Credit amount
                Duration
                Job
    The goal was to reduce de information, by categorizing to numerical al normalizing with z-score
    Finally, there were some outliers to remove
    
    At the end of the code the rest of the code used to visualize data (not required for the final delivery) 
"""

#Libraries 
import pandas as pd
import numpy as np
import joblib as job

#Function for categorizing to numerical data
def categorize(data):
    df = data.copy()
    df['Sex'] = df["Sex"].replace(['male','female'], [1,0])
    df['Saving accounts'] = df["Saving accounts"].replace(['n/a','little','moderate','quite rich','rich'], [0,1,2,3,4])
    df['Checking account'] = df["Checking account"].replace(['n/a','little','moderate','rich'], [0,1,2,3])
    df['Housing'] = df["Housing"].replace(["free","rent","own"],[0,1,2])
    df['Purpose'] = (df["Purpose"].replace(["car","radio/TV","furniture/equipment","business","education","repairs",
                                      "domestic appliances", "vacation/others"],[0,1,2,3,4,5,6,7]))
    df['Risk'] = df["Risk"].replace(['bad','good'], [0,1])
    df["Credit category"] = df["Credit category"].replace(['low', 'moderate', 'high', 'risky'], [0,1,2,3])
    return df

#Function for substracting outliers
def remove_outliers(df, column , min, max):
  outlier = df[(df[column] >= max)|(df[column] <= min)].index
  df.drop(outlier, inplace=True)
  return df

#Function standarizing data with "z-score"
def z_score(feature):
    new_feature = (feature - feature.mean()) / feature.std()
    return new_feature

#Varaible for the file path
path = '../Dataset/Riesgo Crediticio.csv'

#Variable for saving the file and converting 'null' values as 'NaN'
df = pd.read_csv(path, na_values =['NA'])

#Dropping "Unnamed 0" which no represent any parameter
df.drop(['Unnamed: 0'], axis=1, inplace=True)

#Classifying "Credit amount' as: 'low','moderate','high','risky'
risk_type=['low','moderate','high','risky']
credit= pd.cut(df["Credit amount"], bins=len(risk_type), labels=risk_type)
df['Credit category']= pd.Series.to_frame(credit) 

#Bins for "Credit ammount"
bins_credit = np.linspace(min(df["Credit amount"]), max(df["Credit amount"]), len(risk_type)+1)
credit_max = {}
credit_min = {}
credit_type ={}
i=0
for risk in risk_type:
    credit_type[i] = risk
    credit_min[risk] = bins_credit[i]
    i += 1
    credit_max[risk] = bins_credit[i]
credit={"type": credit_type ,"max":credit_max, "min": credit_min}

#Adding 'n/a' category to fill 'Checking account' and 'Saving accounts'
df["Checking account"] = df["Checking account"].fillna("n/a")
df["Saving accounts"] = df["Saving accounts"].fillna("n/a")

#Categorizing data
df = categorize(df)

#Obtening "mean" and "std" from the parameters
mean={}
std={}
for column in (df.columns):
    mean[column] = df[column].mean()
    std[column] =df[column].std()
params={"means":mean,"stds":std, "credit":credit}

#Standarizing data
for column in (df.columns):
  if column != 'Credit category': df[column]= z_score(df[column])
  
#Substracting the outliers
df = remove_outliers(df=df, column='Age' , min = df["Age"].min(), max=3)
df = remove_outliers(df=df, column='Duration', min = df["Duration"].min(), max=4)
df = remove_outliers(df=df, column='Credit amount' , min= df["Credit amount"].min(), max=5) 

#Indexing properly 
orden_id = pd.Series(range(1, df.shape[0]+1 ))
df = df.set_index(orden_id)

#Dropping columns that user must not provide: 'Risk' y 'Credit amount'
df = df.drop(columns=['Risk', 'Credit amount'])

#Saving the new dataset on a new file
df.to_csv('../Dataset/CreditRiskETL.csv')

#Publication of the required paramets
job.dump(params, '../ML_Model/params_dt.joblib')


""" Not used code for this deliverable (stadistical information and data visualization)"""
#Instalation of "Pandas_profiling"
# ! pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

#Libraries not used
# import pandas_profiling
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn import metrics as sk_metrics
# from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler
# from google.colab import drive
# %matplotlib inline

#Function for making a Correlation Matrix
# def make_mc(target, model, name): 
#     labels_Y = []
#     labels = target.value_counts().index
#     for i in labels:
#         labels_Y.append(i)
#     confusion_matrix = sk_metrics.confusion_matrix(Y, model.predict(X))
#     df_confusion_matrix = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
#     plt.figure(figsize=(8, 8))
#     sns.heatmap(df_confusion_matrix, annot=True, cbar=False, cmap='Oranges', linewidths=1, linecolor='black')
#     plt.xlabel('Etiquetas predichas', fontsize=15)
#     plt.xticks(fontsize=16)
#     plt.ylabel('Etiquetas verdaderas', fontsize=15)
#     plt.yticks(fontsize=16)
#     plt.title(name)

#Function for categorizing to numerical data (other way- not recommended)
# def categorize(data):
#     df = data.copy()
#     LE = preprocessing.LabelEncoder()
#     df['Sex'] = LE.fit_transform(df['Sex'])
#     df['Housing'] = LE.fit_transform(df['Housing'])
#     df['Saving accounts'] = LE.fit_transform(df['Saving accounts'])
#     df['Checking account'] = LE.fit_transform(df['Checking account'])
#     df['Purpose'] = LE.fit_transform(df['Purpose'])
#     df['Risk'] = LE.fit_transform(df['Risk'])
#     df["Credit category"] = df["Credit category"].replace(['low', 'moderate', 'high', 'risky'], [0,1,2,3])
#     return df

#Function for calculating the Information Value (IV)
# def calc_iv(df, feature, target):
#   lst = []
#   for i in range(df[feature].nunique()):
#     val = list(df[feature].unique())[i]
#     lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])
#   data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Bad'])
#   data = data[data['Bad'] > 0]
#   data['Share'] = data['All'] / data['All'].sum()
#   data['Bad Rate'] = data['Bad'] / data['All']
#   data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
#   data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
#   data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
#   data['IV'] = (data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])).sum()
#   data = data.sort_values(by=['Variable', 'Value'], ascending=True)
#   return data['IV'].values[0]

#Function for standarizing the data with "simple feature scaling"
# def simple_feature_scaling(feature):
#     new_feature = feature / feature.max()
#     return new_feature

#Function for standarizing the data with "min max scalinng"
# def min_max_scaling(feature):
#     new_feature = (feature - feature.min()) / (feature.max() - feature.min())
#     return new_feature

#Using Google Drive as the main folder
# drive.mount('/content/drive/')

#Counting null data
# df.isnull().sum()

#Reporting the correlation of the data
#pandas_profiling.ProfileReport(df)

#Obtainin the shape of the df
# df.shape

#Visualizing the information
#Age
# sns.set(style="ticks", color_codes=True)
# sns.distplot(df['Age'])
#Duration
# sns.set(style="ticks", color_codes=True)
# sns.distplot(df['Duration'])
#Credit amount
# sns.set(style="ticks")
# sns.distplot(df["Credit amount"])
#Sex
# sns.set(style="ticks")
# sns.countplot(x='Sex', data=df)
#Job
# sns.set(style="ticks")
# sns.countplot(x='Job', data=df)
#Housing
# sns.set(style="ticks")
# sns.countplot(x='Housing', data=df)
#Saving Accounts
# sns.set(style="ticks")
# sns.countplot(x='Saving accounts', data=df)
#Checking Account
# sns.set(style="ticks")
# sns.countplot(x='Checking account', data=df)
#Purpose
# sns.set(style="ticks")
# sns.countplot(x='Purpose', data=df)
#Risk
# sns.set(style="ticks")
# sns.countplot(x='Risk', data=df)
#Credit ammount category
# sns.set(style="ticks")
# sns.countplot(x='Credit category', data=df)
#Age category
#sns.set(style="ticks")
#sns.countplot(x='Age category', data=df)
#Duration category
#sns.set(style="ticks")
#sns.countplot(x='Duration category', data=df)


#Observing the range: 'Credit amount category', 'Age category', 'Duration category'
#Credit amount category
# class_credit= pd.cut(df["Credit amount"], bins=len(risk_type)).value_counts()
# class_credit
#Age category
#clases_age= pd.cut(df["Age"], bins=len(age_range)).value_counts()
#class_age
#Duration category
#class_duration= pd.cut(df["Duration"], bins=len(duration_range)).value_counts()
#class_duration

#Describing stadistically the data
# df.describe(include='all')
