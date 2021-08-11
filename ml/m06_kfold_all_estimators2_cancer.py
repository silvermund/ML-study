# 실습, 모델 구성하고 완료하시오.
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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)


#1. 데이터
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(569, 30) (569,)

print(y[:20]) # y가 0과 1인, 2진 분류
print(np.unique(y))


# 데이터 전처리
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

# scaler = StandardScaler()
# scaler.fit(x_train) # 훈련
# x_train = scaler.transform(x_train) # 변환
# x_test = scaler.transform(x_test)
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms)
print("모델의 갯수 : ", len(allAlgorithms)) #41

for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()

        scores = cross_val_score(model, x, y, cv=kfold)

        
        print(name, scores)
    except:
        # continue
        print(name, 'is N/A')


# model = SVC()
# 0.9766081871345029

# model = LinearSVC()
# 0.9766081871345029

# model = KNeighborsClassifier()
# accuracy_score :  0.9590643274853801

# model = KNeighborsRegressor()

# model = LogisticRegression()
# accuracy_score :  0.9824561403508771

# model = RandomForestClassifier()
# accuracy_score :  0.9590643274853801

# model = DecisionTreeClassifier()
# accuracy_score :  0.9415204678362573


# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(30,))) 
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
# model.fit(x_train, y_train)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

# hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es])

# print(hist.history.keys())
# print("================================")
# print(hist.history['loss'])
# print("================================")
# print(hist.history['val_loss'])
# print("================================")

#4. 평가, 예측
# results = model.score(x_test, y_test)
# print(results)
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)



# print("=================예측===============")
# print(y_test[:5])
# y_predict2 = model.predict(x_test[:5])
# print(y_predict2)

# print("==============평가, 예측=============")
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss[0])
# print('accuracy : ', loss[1])

# print("=================예측===============")
# print(y_test[-5:-1])
# y_predict = model.predict(x_test[-5:-1])
# print(y_predict)

# y_predict = model.predict(x_test)
# # print('x_predict의 예측값 : ', y_predict)

# r2 = r2_score(y_test, y_predict)
# print("r2스코어 :", r2)

# plt.plot(hist.history['loss'])  # x:epoch, / y:hist.history['loss']
# plt.plot(hist.history['val_loss'])

# plt.title("loss, val_loss")
# plt.xlabel('epoch')
# plt.ylabel('loss, val_loss')
# plt.legend(['train loss','val loss'])
# plt.show()

'''
loss :  0.014620478264987469
r2스코어 : 0.9362865232268573

binary_crossentropy
loss :  0.45230045914649963
r2스코어 : 0.8962701901084006

loss: 2.6394e-09 - accuracy: 1.0000

'''

# 모델의 갯수 :  41
# AdaBoostClassifier [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133]
# BaggingClassifier [0.94736842 0.92105263 0.94736842 0.92105263 0.95575221]
# BernoulliNB [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858]
# CalibratedClassifierCV [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133]
# CategoricalNB [nan nan nan nan nan]
# ClassifierChain is N/A
# ComplementNB [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531]
# DecisionTreeClassifier [0.93859649 0.93859649 0.93859649 0.87719298 0.95575221]
# DummyClassifier [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858]
# ExtraTreeClassifier [0.93859649 0.9122807  0.88596491 0.9122807  0.91150442]
# ExtraTreesClassifier [0.96491228 0.98245614 0.96491228 0.94736842 0.98230088]
# GaussianNB [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221]
# GaussianProcessClassifier [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265]
# GradientBoostingClassifier [0.94736842 0.97368421 0.95614035 0.93859649 0.98230088]
# HistGradientBoostingClassifier [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088]
# KNeighborsClassifier [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221]
# LabelPropagation [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053]
# LabelSpreading [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053]
# LinearDiscriminantAnalysis [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133]
# LinearSVC [0.92105263 0.93859649 0.87719298 0.87719298 0.97345133]
# LogisticRegression [0.93859649 0.95614035 0.88596491 0.95614035 0.96460177]
# LogisticRegressionCV [0.95614035 0.97368421 0.9122807  0.96491228 0.96460177]
# MLPClassifier [0.90350877 0.93859649 0.92105263 0.9122807  0.94690265]
# MultiOutputClassifier is N/A
# MultinomialNB [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531]
# NearestCentroid [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442]
# NuSVC [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575]
# OneVsOneClassifier is N/A
# OneVsRestClassifier is N/A
# OutputCodeClassifier is N/A
# PassiveAggressiveClassifier [0.90350877 0.94736842 0.88596491 0.89473684 0.97345133]
# Perceptron [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265]
# QuadraticDiscriminantAnalysis [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265]
# RadiusNeighborsClassifier [nan nan nan nan nan]
# RandomForestClassifier [0.96491228 0.95614035 0.97368421 0.94736842 0.98230088]
# RidgeClassifier [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221]
# RidgeClassifierCV [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177]
# SGDClassifier [0.81578947 0.76315789 0.85964912 0.80701754 0.92035398]
# SVC [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177]
# StackingClassifier is N/A
# VotingClassifier is N/A