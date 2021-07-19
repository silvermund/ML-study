from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv1D, Flatten, MaxPool1D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
import time

datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)


#1. 데이터
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(569, 30) (569,)

# print(y[:20]) # y가 0과 1인, 2진 분류
# print(np.unique(y))


# 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

scaler = StandardScaler()
scaler.fit(x_train) # 훈련
x_train = scaler.transform(x_train) # 변환
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape) #(398, 30, 1)
print(x_test.shape) #(171, 30, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=8, kernel_size=2, padding='same', input_shape=(30,1)))
model.add(Dropout(0.2))
model.add(Conv1D(8, 2, padding='same', activation='relu'))   
model.add(MaxPool1D())

model.add(Conv1D(32, 2, padding='same', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, padding='same', activation='relu')) 
model.add(MaxPool1D())

model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, padding='same', activation='relu'))  
model.add(MaxPool1D())

# model.add(Flatten())    
# model.add(Dense(128, activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))  
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu')) 
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation="sigmoid"))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)


start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=4, validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time


#4. 평가, 예측
y_predict = model.predict([x_test])

loss = model.evaluate(x_test, y_test)
print("time = ", end_time)
print('loss : ', loss)
r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)



'''
dnn
# loss: 2.6394e-09 - accuracy: 1.0000

cnn
time =  46.88242268562317
loss :  0.02056293934583664
R^2 score :  0.9103903276773451
'''
