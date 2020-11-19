
"""
    This code help us to train an publish our ML model
    The best algorithm found was Decision Tree with a 85-15 data split
    A GridSearch was applied in order to find the best solution
    The two target paramerts were: Accuracy and AUC
    
    At the end of the code the rest of the algorithms trained but not choosen 
"""

#Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import joblib as job

#Fuction for getting the AUC
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

#Data aquisistion
path = '../Dataset/CreditRiskETL.csv'
df = pd.read_csv(path)
df = df.drop('Unnamed: 0', axis=1)

#Division of data to entries (X) and output (Y)-'Credit Category'
X = df.drop(['Credit category', 'Job'],axis = 1).values
Y = df['Credit category'].values

#Split of data into 'Train' (0.85) and 'Test' (0.15)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

#GridSearch for the algorithm
dt_params = {'criterion':['gini','entropy'],
             'splitter':['best','random']}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42),dt_params, verbose=0,cv=9,n_jobs=6,scoring='accuracy')
dt_grid.fit(X_train, Y_train)

#Information of 'Trainset' & 'Testset' estimations
dt = dt_grid.best_estimator_
accuracy_train= dt_grid.best_score_
accuracy_test = dt.score(X_test,Y_test)
auc_train = multiclass_roc_auc_score(Y_train, dt.predict(X_train))
auc_test = multiclass_roc_auc_score(Y_test, dt.predict(X_test))
print(f"Train info DT: accuracy = {accuracy_train},auc ={auc_train}") 
print(f"Test info DT:accuracy = {accuracy_test},auc ={auc_test}") 

#Prediction
array = np.array([X_train[0]])
prediction = dt.predict(array.reshape(1, -1))
print(f"Predition:{prediction}")

#Publication of the model
job.dump(dt, 'model_dt.joblib')




""""Standarization no required because data included z-score"""
#from sklearn.preprocessing import StandardScaler
#StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.fit_transform(X_train)
# scaler.fit(X_test)
# X_test = scaler.fit_transform(X_test)

"""Other algorithms with lower ACU"""
#Libraries
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier

#KNN
# knn_params = {'n_neighbors':[1,2,4,5,6,7,8,9],
#               'weights':['uniform', 'distance'],
#               'algorithm':['auto','ball_tree','kd_tree','brute'], 
#               'p':[1,2,3]}
# knn_grid = GridSearchCV(KNeighborsClassifier(),knn_params, verbose=0,cv=9,n_jobs=6,scoring='accuracy')
# knn_grid.fit(X_train, Y_train)

#GaussianNB
# gauss_params = {'var_smoothing':[1e-09,1e-11,1e-7,1e-5,1e-13]}

# gauss_grid = GridSearchCV(GaussianNB(),gauss_params, verbose=0,cv=9,n_jobs=6,scoring='accuracy')
# gauss_grid.fit(X_train, Y_train)
# # print(dt_grid.best_estimator_)

#SVM
# svm_params = {'C':[0.001,0.1,1],
#               'kernel':['linear','poly','rbf', 'sigmoid'],
#               'decision_function_shape':['ovo','ovr'],
#               'degree':[3,5],
#               'gamma':['scale','auto']}
# svm_grid = GridSearchCV(SVC(),svm_params, verbose=0,cv=9,n_jobs=4,scoring='accuracy')
# svm_grid.fit(X_train, Y_train)

#Random Forest
# rf_params = {'criterion':['gini','entropy'],
#               'n_estimators':list(range(10,201,10))}
# rf_grid = GridSearchCV(RandomForestClassifier(),rf_params, verbose=0,cv=9,n_jobs=4,scoring='accuracy')
# rf_grid.fit(X_train, Y_train)

#Logistic Regression
# lr_params = {'penalty':['l1','l2','elasticnet','none'],
#              'C':[0.001,0.1,1],
#              'tol':[1e-4,1e-6],
#              'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#              'multi_class':['auto', 'ovr', 'multinomial']}
# lr_grid = GridSearchCV(LogisticRegression(),lr_params, verbose=0,cv=9,n_jobs=6,scoring='accuracy')
# lr_grid.fit(X_train, Y_train)

#Multi Layer Perceptron
# mlp_params = {'activation':['identity', 'logistic', 'tanh', 'relu'], 
#               'alpha':[0.0001,0.000001],     
#               'hidden_layer_sizes':[(8,8),(8,8,8),(100,)],
#               'solver':['lbfgs', 'sgd', 'adam'],
#               'max_iter':[200],
#               'learning_rate':['constant', 'invscaling', 'adaptive']}
# mlp_grid = GridSearchCV(MLPClassifier(),mlp_params, verbose=1,cv=9,n_jobs=4,scoring='accuracy')
# mlp_grid.fit(X_train, Y_train)

