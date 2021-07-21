from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(442,10) (442,)

print(datasets.feature_names) #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=5)

#2. 모델구성
model = Sequential()
model.add(Dense(2048, input_dim=10)) 
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.save('./_save/keras46_1_save_model_1.h5') #모델만 저장하고 싶을 때


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

model.save('./_save/keras46_1_save_model_2.h5') #가중치까지 저장하고 싶을 때


#4. 평가, 예측
# mse, R2
print('걸린 시간 : ', end_time)
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('x_predict의 예측값 : ', y_predict)

r2 = r2_score(y, y_predict)

print("r2스코어 :", r2)


'''
r2스코어 : 0.586778592520653
'''

