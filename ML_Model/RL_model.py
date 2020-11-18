
"""##4. Modelado de la Información

Una vez finalizada limpieza general del dataset y extracción de la información, se busca seleccionar el mejor algoritmo de aprendizaje automático. Por tal motivo, a cotninuación, se hace una comparación del 'accuracy' con cada uno de los distintos algoritmos (algunos de ellos vistos en teoría).  

Los algoritmos que se plantean tienen son utilizados de manera general para la tarea cde clasificación, misma que se tiene como objetivo en nuestro proyecto. Esta sección nos ayudará a evaluar y optimizar el mejor (o los mejores, en caso de que los resultados sean muy similares) para generar el mejor 'accuracy' posible para el dataset y el problema planteado.

###4.1. Construcción de 'Trainset' y 'Testset'
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib as job

path = '../Dataset/CreditRiskETL.csv'
df = pd.read_csv(path)
df = df.drop('Unnamed: 0', axis=1)

#Partición de las variables de entrada (X) y la variable de salida (Y)-'Credit Category'
X = df.drop(['Credit category'],axis = 1).values
Y = df['Credit category'].values

#Disvisión de la información 80-20, 'train' y 'test',
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

#Normalización de información para los algoritmos de M.L.
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.fit_transform(X_train)
# scaler.fit(X_test)
# X_test = scaler.fit_transform(X_test)
# print(X_train[0])

dt_params = {'criterion':['gini','entropy'],
             'splitter':['best','random']}

dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42),dt_params, verbose=0,cv=9,n_jobs=8,scoring='accuracy')
dt_grid.fit(X_train, Y_train)
# print(dt_grid.best_estimator_)

#Accuracy con 'Trainset' & 'Testset'
dt = dt_grid.best_estimator_
accuracy_train= dt_grid.best_score_
accuracy_test = dt.score(X_test,Y_test)
print(f"Train accuracy:{accuracy_train}") 
print(f"Test accuracy:{accuracy_test}") 

# Prediction
array = np.array([2.67716444, 0.66994484, 1.67683077, -0.13364358, -0.70614255,  1.8910083, 0.75438593, -1.45490372])
prediction = dt.predict(array.reshape(1, -1))
print(f"Predition:{prediction}")
job.dump(dt, 'model_dt.joblib')
