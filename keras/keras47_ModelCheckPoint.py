from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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


model.summary()

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath='./_save/ModelCheckPoint/keras47_MCP.hdf5')

start_time = time.time()
model.fit(x_train, y_train, epochs=30, batch_size=1, validation_split=0.2, callbacks=[es, cp])
end_time = time.time() - start_time


model.save('./_save/ModelCheckPoint/keras47_model_save.hdf5')

#4. 평가, 예측
loss = model.evaluate(x, y)
y_predict = model.predict(x)
r2 = r2_score(y, y_predict)
print('걸린 시간 : ', end_time)
print('loss : ', loss)
print("r2스코어 :", r2)

'''
걸린 시간 :  6.4389307498931885
loss :  3200.939208984375
r2스코어 : 0.460202108959304
'''

