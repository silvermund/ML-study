from sklearn.utils import all_estimators

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
# from sklearn import datasets
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)


np = datasets.values


x = np[:,:11]
y = np[:,11:]

ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)

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

# parameters =[
#     {'n_jobs' : [-1], 'n_estimators' : [100,200], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [5, 7, 10]},
#     {'n_jobs' : [-1], 'max_depth' : [5,6,7], 'min_samples_leaf' : [6,7,11], 'min_samples_split' : [3,4,5]},
#     {'n_jobs' : [-1], 'min_samples_leaf' : [3,5,7], 'min_samples_split' : [2,3,5,10]},
#     {'n_jobs' : [-1], 'min_samples_split' : [2,3,5,10]},
# ]
parameters =[
    {'randomforestclassifier__min_samples_leaf' : [3, 5, 7], 'randomforestclassifier__max_depth' : [2, 3, 5, 10]},
    {'randomforestclassifier__min_samples_split' : [6, 8, 10]},
]


# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)
# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold)
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier()) 

model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)


start_time = time.time()
model.fit(x_train, y_train)

# print("최적의 매개변수 : ", model.best_estimator_)
# print("best_score_ : ", model.best_score_)

print("model.score: ", model.score(x_test, y_test))

y_predict = model.predict(x_test)

print("accuracy_score : ", accuracy_score(y_test, y_predict))

print("r2_score : ", r2_score(y_test, y_predict))

print("걸린시간 : ", time.time() - start_time)


# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=-1)
# best_score_ :  0.5043787108169651
# model.score:  0.5517006802721088
# 정답률 :  0.5517006802721088
# 걸린시간 :  70.13569664955139

# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=3, min_samples_split=5, n_jobs=-1)
# best_score_ :  0.47520631610308356
# model.score:  0.5054421768707483
# 정답률 :  0.5054421768707483
# 걸린시간 :  14.938013076782227


# pipeline
# model.score:  0.5523809523809524
# 정답률 :  0.5523809523809524
# 걸린시간 :  0.8319988250732422

# Fitting 5 folds for each of 15 candidates, totalling 75 fits
# model.score:  0.5319727891156463
# accuracy_score :  0.5319727891156463
# r2_score :  0.05063841128933485
# 걸린시간 :  28.598999977111816