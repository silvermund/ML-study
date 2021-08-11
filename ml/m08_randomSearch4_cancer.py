from sklearn.utils import all_estimators

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
# from sklearn import datasets
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

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

# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)

start_time = time.time()
model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
print("best_score_ : ", model.best_score_)

print("model.score: ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("정답률 : ", accuracy_score(y_test, y_predict))

print("걸린시간 : ", time.time() - start_time)



# Fitting 5 folds for each of 61 candidates, totalling 305 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=5, n_jobs=-1)
# best_score_ :  0.9647784810126583
# model.score:  0.9649122807017544
# 정답률 :  0.9649122807017544
# 걸린시간 :  45.76800179481506

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=5, min_samples_split=10, n_jobs=-1)
# best_score_ :  0.9573417721518988
# model.score:  0.9649122807017544
# 정답률 :  0.9649122807017544
# 걸린시간 :  10.166974544525146

