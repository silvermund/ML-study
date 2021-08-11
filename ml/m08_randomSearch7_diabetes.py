from sklearn.utils import all_estimators

from sklearn.svm import LinearSVC, SVC # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=5)


n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)



# parameters =[
#     {'n_estimators' : [100, 200]},
#     {'max_depth' : [6, 8, 10, 12]},
#     {'min_samples_leaf' : [3, 5, 7, 10]},
#     {'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]}
# ]

parameters =[
    {'n_jobs' : [-1], 'n_estimators' : [100,200], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [5, 7, 10]},
    {'n_jobs' : [-1], 'max_depth' : [5,6,7], 'min_samples_leaf' : [6,7,11], 'min_samples_split' : [3,4,5]},
    {'n_jobs' : [-1], 'min_samples_leaf' : [3,5,7], 'min_samples_split' : [2,3,5,10]},
    {'n_jobs' : [-1], 'min_samples_split' : [2,3,5,10]},
]



model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1)
# model =RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1)

start_time = time.time()
model.fit(x_train, y_train)


# print("최적의 매개변수 : ", model.best_estimator_)
# print("best_params_ : ", model.best_params_)
# print("best_score_ : ", model.best_score_)

print("model.score: ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("r2_score : ", r2_score(y_test, y_predict))

print("걸린시간 : ", time.time() - start_time)

# GridSearchCV
# Fitting 5 folds for each of 61 candidates, totalling 305 fits
# model.score:  0.5413766369870819
# r2_score :  0.541376636987082
# 걸린시간 :  42.724019289016724


# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# model.score:  0.533134489688982
# r2_score :  0.533134489688982
# 걸린시간 :  10.284970760345459