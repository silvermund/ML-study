from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv2D, Flatten, MaxPool2D, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
import time

# 실습
# 1~100까지의 데이터를 

#       x              y
# 1, 2, 3, 4, 5,       6
# ...
# 95, 96, 97, 98,99   100


#1. 데이터

x_data = np.array(range(1, 101))
# x_data로 x와 y를 만들것 

x_predict = np.array(range(96, 105))
#              x
# 96, 97, 98, 99, 100       ?
# ...
# 101, 102, 103, 104, 105   ?

# 예상 결과값 : 101 102 103 104 105 106


size1 = 6
size2 = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size1)
x_predict = split_x(x_predict, size2)

# print(dataset)

x = dataset[:, :-1]
y = dataset[:, -1]
x_predict = x_predict[:, :-1]

# print("x : \n", x)
# print("y : ", y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)

# en = OneHotEncoder()
# y_train = en.fit_transform(y_train).toarray()
# y_test = en.fit_transform(y_test).toarray()

print(x.shape, y.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# print(x_train)
# print(y_train)

#2. 모델구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(5,1)))
model.add(Dense(9, activation='relu'))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

#4. 평가, 예측
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

r2 = r2_score(y_test, y_predict)
print("==============평가, 예측==============")
print('rmse score : ', rmse)
print('걸린 시간 : ', end_time)
print("r2스코어 :", r2)

'''
rmse score :  5.6713620287637205
걸린 시간 :  8.062105655670166
r2스코어 : 0.9516381813120058
'''