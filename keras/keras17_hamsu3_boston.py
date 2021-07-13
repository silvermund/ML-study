# 보스턴을 함수형을 구현하시오.
# 서머리 확인

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


# 완료하시오!!!
# 0709과제

datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=5)

print(x.shape)
print(y.shape)
print(datasets.feature_names) ## ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] 13의 요소
print(datasets.DESCR)


input1 = Input(shape=(13,))
dense1 = Dense(55)(input1)
dense2 = Dense(44)(dense1)
dense3 = Dense(33)(dense2)
dense4 = Dense(22)(dense3)
dense5 = Dense(11)(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)
model.summary()

# model = Sequential()
# model.add(Dense(55, input_dim=13)) #위의 종류만큼 input
# model.add(Dense(44))
# model.add(Dense(33))
# model.add(Dense(22))
# model.add(Dense(11))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=100, batch_size=1)

loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('x_predict의 예측값 : ', y_predict)

r2 = r2_score(y, y_predict)

print("r2스코어 :", r2)
