from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv1D, Flatten, MaxPool1D, Dropout, GlobalAveragePooling1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
import time


datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(442,10) (442,)

# print(datasets.feature_names) #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=5)

print(x_train.shape) #(309, 10)
print(x_test.shape) #(133, 10)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 변환
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape) #(309, 10, 1)
print(x_test.shape) #(133, 10, 1)


#2. 모델구성
model = Sequential()
# model.add(Conv1D(filters=16, kernel_size=2, padding='same', input_shape=(10,1)))
# model.add(Dropout(0.2))
# model.add(Conv1D(16, 2, padding='same', activation='relu'))   
# model.add(MaxPool1D())

# model.add(Conv1D(64, 2, padding='same', activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Conv1D(64, 2, padding='same', activation='relu')) 
# model.add(MaxPool1D())

# model.add(Conv1D(256, 2, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv1D(256, 2, padding='same', activation='relu'))  
# model.add(MaxPool1D())

# model.add(GlobalAveragePooling1D())
# model.add(Dense(1))

model.add(LSTM(16, activation='relu', input_length=10, input_dim=1))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)


start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.25, callbacks=[es])
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
r2스코어 : 0.586778592520653

cnn
time =  8.976810932159424
loss :  3904.624755859375
R^2 score :  0.4007979466926965

lstm
time =  63.351191997528076
loss :  4817.64794921875
R^2 score :  0.2606857955287666
'''