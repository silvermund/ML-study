from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate
import numpy as np
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler

datasets = load_boston()

#1. 데이터
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x)) # 0.0 711.0

# 데이터 전처리
# x = (x - np.min(x)) / (np.max(x) - np.min(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

# print(x_scale.shape) # (506, 13)
# print(x_train.shape) # (354, 13)
# print(x_test.shape) # (152, 13)
# print(datasets.feature_names) ## ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] 13의 요소
# print(datasets.DESCR)

scaler = MinMaxScaler()
scaler.fit(x_train) # 훈련
x_train = scaler.transform(x_train) # 변환
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,))) 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# print('x_predict의 예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict)
print("r2스코어 :", r2)


'''
전처리 전
loss :  16.859941482543945
r2스코어 : 0.7959267298884826

minmax 전처리 후
loss :  12.270087242126465
r2스코어 : 0.8514824691193794

MinMaxScalar 후
loss :  6.314334392547607
r2스코어 : 0.9235710971311957

MinMaxScalar, transform 후
loss :  5.841597080230713
r2스코어 : 0.9292931182133031
'''


