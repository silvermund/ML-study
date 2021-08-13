# m31로 만든 0.999 이상의 n_component=?를 사용하여
# xgb 모델을 만들 것 (디폴트)

# mnist dnn 보다 성능 좋게 만들어라!!
# dnn, cnn과 비교!!

# RandomSearch 로도 해볼것

parameters = [
    {"n_estimators":[90, 100, 110, 200, 300], 
    "learning_rate":[0.001, 0.01],
    "max_depth":[4, 5, 6], 
    "colsample_bytree":[0.6, 0.9, 1], 
    "colsample_bylevel":[0.6, 0.7, 0.9],
    "n_jobs":[-1]}
]

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import time
import warnings 
warnings.filterwarnings(action='ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA(n_components=154)
x = pca.fit_transform(x)


#print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

# x = np.append(x_train, x_test, axis=0)
# print(x.shape) #(70000, 28, 28)
# x = x.reshape(70000, 28*28)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 변환
x_test = scaler.transform(x_test)

# en = OneHotEncoder()
# y_train = en.fit_transform(y_train).toarray()
# y_test = en.fit_transform(y_test).toarray()


model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1)


start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

print('totla time : ', end_time)
print('Best estimator : ', model.best_estimator_)
print('Best score  :', model.best_score_)

# Best estimator :  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
#               colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.01, max_delta_step=0, max_depth=5,
#               min_child_weight=1, missing=nan, monotone_constraints='()',
#               n_estimators=300, n_jobs=-1, num_parallel_tree=1,
#               objective='multi:softprob', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='exact', validate_parameters=1, verbosity=None)
# Best score  : 0.9016964285714286