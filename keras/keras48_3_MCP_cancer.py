from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv1D, Flatten, MaxPool1D, Dropout, GlobalAveragePooling1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
# model.add(Conv1D(filters=8, kernel_size=2, padding='same', input_shape=(30,1)))
# model.add(Dropout(0.2))
# model.add(Conv1D(8, 2, padding='same', activation='relu'))   
# model.add(MaxPool1D())

# model.add(Conv1D(32, 2, padding='same', activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Conv1D(32, 2, padding='same', activation='relu')) 
# model.add(MaxPool1D())

# model.add(Conv1D(128, 2, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv1D(128, 2, padding='same', activation='relu'))  
# model.add(MaxPool1D())
# model.add(GlobalAveragePooling1D())
# model.add(Dense(1, activation="sigmoid"))

# model.add(LSTM(16, activation='relu', input_length=30, input_dim=1))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1))

model.add(Conv1D(64, 2, input_shape=(30, 1)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath='./_save/ModelCheckPoint/keras48_3_MCP.hdf5')


start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=4, validation_split=0.25, callbacks=[es, cp])
end_time = time.time() - start_time


# model.save('./_save/ModelCheckPoint/keras48_3_model_save.hdf5')



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

lstm
time =  848.1184375286102
loss :  0.018325380980968475
R^2 score :  0.9201412120814718

conv1d
time =  15.137083053588867
loss :  0.05595207214355469
R^2 score :  0.756170736987745

model
time =  16.493212938308716
loss :  0.056137554347515106
R^2 score :  0.7553624005733481

MCP
time =  17.61366367340088
loss :  0.05339721217751503
R^2 score :  0.7673043291188956
'''
