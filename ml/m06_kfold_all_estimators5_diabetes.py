# 실습, 모델 구성하고 완료하시오.
# 회귀 데이터를 Classifier로 만들었을 경우에 에러 확인!!!
from sklearn.utils import all_estimators

from sklearn.svm import LinearSVC, SVC # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 실습 diabetes
# 1. loss와 R2로 평가를 함
# MinMax와 Standard 결과를 명시

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

datasets = load_diabetes()

#1. 데이터
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(442,10) (442,)

print(datasets.feature_names) #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))

# 데이터 전처리
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=5)
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# scaler = MinMaxScaler()
# scaler.fit(x_train) # 훈련
# x_train = scaler.transform(x_train) # 변환
# x_test = scaler.transform(x_test)


#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms)
print("모델의 갯수 : ", len(allAlgorithms)) #41

for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()

        scores = cross_val_score(model, x, y, cv=kfold)

        # model.fit(x_train, y_train)

        # y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        print(name, scores)
    except:
        # continue
        print(name, 'is N/A')

# model = LinearSVC()
# accuracy_score :  0.0

# model = SVC()
# accuracy_score :  0.02247191011235955

# model = KNeighborsClassifier()

# model = KNeighborsRegressor()

# model = LogisticRegression()
# accuracy_score :  0.011235955056179775

# model = RandomForestClassifier()
# accuracy_score :  0.011235955056179775

# model = DecisionTreeClassifier()
# accuracy_score :  0.0

# input1 = Input(shape=(10,))
# dense1 = Dense(64)(input1)
# dense2 = Dense(32, activation='relu')(dense1)
# dense3 = Dense(32, activation='relu')(dense2)
# dense4 = Dense(32, activation='relu')(dense3)
# dense5 = Dense(32, activation='relu')(dense4)
# dense6 = Dense(32, activation='relu')(dense5)
# dense7 = Dense(8, activation='relu')(dense6)
# output1 = Dense(1)(dense7)

# model = Model(inputs=input1, outputs=output1)
# model.summary()


# model = Sequential()
# model.add(Dense(2048, input_dim=10)) 
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

#3. 컴파일, 훈련
# model.fit(x_train, y_train)
# model.compile(loss='mse', optimizer='adam')

# model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, shuffle=True)

#4. 평가, 예측
# mse, R2
# results = model.score(x_test, y_test)
# print(results)
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)



# print("=================예측===============")
# print(y_test[:5])
# y_predict2 = model.predict(x_test[:5])
# print(y_predict2)


'''
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# print('x_predict의 예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict)

print("r2스코어 :", r2)



MinMaxScaler 선택
loss :  3256.529052734375
r2스코어 : 0.48355337849993907

StandardScaler 선택
loss :  5215.6591796875
r2스코어 : 0.17285872820333992

'''
# 모델의 갯수 :  41
# AdaBoostClassifier [0.         0.         0.         0.01136364 0.        ]
# BaggingClassifier [0.01123596 0.01123596 0.         0.         0.        ]
# BernoulliNB [0.         0.         0.         0.         0.01136364]
# CalibratedClassifierCV [0.01123596 0.03370787 0.         0.         0.01136364]
# CategoricalNB [0. 0. 0. 0. 0.]
# ClassifierChain is N/A
# ComplementNB [nan nan nan nan nan]
# DecisionTreeClassifier [0.         0.         0.01136364 0.         0.01136364]
# DummyClassifier [0. 0. 0. 0. 0.]
# ExtraTreeClassifier [0.02247191 0.         0.         0.         0.        ]
# ExtraTreesClassifier [0.01123596 0.01123596 0.01136364 0.         0.01136364]
# GaussianNB [0.01123596 0.01123596 0.         0.         0.        ]
# GaussianProcessClassifier [0.         0.         0.01136364 0.         0.        ]
# GradientBoostingClassifier [0.         0.         0.01136364 0.         0.        ]