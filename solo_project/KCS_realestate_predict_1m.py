from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv1D, Flatten, MaxPool2D, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
import time
from tensorflow.python.ops.gen_math_ops import Min
plt.rcParams['font.family'] = 'NanumGothic'

#1. 데이터

x1_data = pd.read_csv('./solo_project/real_estate_predict.csv', encoding='EUC-KR', usecols=[1,2,3,4])
x1_data = x1_data[0:2256]
x1_data = np.array(x1_data)
print(x1_data.shape) #(2256, 4)

x2_data = pd.read_csv('./solo_project/real_estate_predict.csv', encoding='EUC-KR', usecols=[5,6,7])
x2_data = x2_data[0:2256]
x2_data = np.array(x2_data)
print(x2_data.shape) #(2256, 3)

y_data = pd.read_csv('./solo_project/real_estate_predict.csv', encoding='EUC-KR', usecols=[8])
y_data = y_data[21:2277]
y_data = np.array(y_data)
print(y_data.shape) #(2256, 1)

size = 5

def split_x(x1, size):
    aaa = []
    for i in range(len(x1) - size + 1):
        subset = x1[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

x1 = split_x(x1_data, size)
print(x1.shape) #(2252, 5, 4)

x1 = x1.reshape(2252*5, 4)

x1 = x1[:2252*5]

def split_x(x2, size):
    aaa = []
    for i in range(len(x2) - size + 1):
        subset = x2[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

x2 = split_x(x2_data, size)
print(x2.shape) #(2252, 5, 3)

x2 = x2.reshape(2252*5, 3)
x2 = x2[:2252*5]

def split_x(y, size):
    aaa = []
    for i in range(len(y) - size + 1):
        subset = y[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

y = split_x(y_data, size)
print(y.shape) #(2252, 5, 1)

y = y.reshape(2252*5, 1)
y = y[:,0]


x1_predict = x1[-5:]
x2_predict = x2[-5:]


print(x1_predict.shape) #(5, 4)
print(x2_predict.shape) #(5, 3)


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, random_state=66) 
print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape) 
# (7881, 4) (3379, 4) (7881, 3) (3379, 3) (7881,) (3379,)

# print(type(x1_train))
# # print('-------')
# print(type(x1_test))
# print(type(y_train))



# scaling
# scaler = MinMaxScaler()
# scaler.fit_transform(x1_train)
# scaler.fit_transform(x2_train)
# scaler.transform(x1_test)
# scaler.transform(x2_test)
# scaler.transform(x1_predict)
# scaler.transform(x2_predict)

#2-1. 모델1
input1 = Input(shape=(4,1))
xx = LSTM(units=128, activation='relu', return_sequences=True)(input1)
xx = Conv1D(32,2, activation='relu')(xx)
xx = LSTM(units=32, activation='relu', return_sequences=True)(xx)
xx = Conv1D(16,2, activation='relu')(xx)
xx = LSTM(units=4, activation='relu', return_sequences=True)(xx)
xx = LSTM(units=2, activation='relu', return_sequences=True)(xx)
xx = Flatten()(xx)
xx = Dense(128, activation='relu')(xx)
xx = Dropout(0.2)(xx)
output1 = Dense(16, name='output1', activation='relu')(xx)

# 2-2 model2
input2 = Input(shape=(3,1))
xx = LSTM(units=128, activation='relu', return_sequences=True)(input1)
xx = Conv1D(32,2, activation='relu')(xx)
xx = LSTM(units=32, activation='relu', return_sequences=True)(xx)
xx = Conv1D(16,2, activation='relu')(xx)
xx = LSTM(units=4, activation='relu', return_sequences=True)(xx)
xx = LSTM(units=2, activation='relu', return_sequences=True)(xx)
xx = Flatten()(xx)
xx = Dense(128, activation='relu')(xx)
xx = Dropout(0.2)(xx)
output2 = Dense(16, name='output2', activation='relu')(xx)

merge1 = concatenate([output1, output2]) 
merge2 = Dense(128, activation='relu')(merge1)
merge3 = Dense(16, activation='relu')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)

###########################################################################################################
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './samsung/_save/ModelCheckPoint/'
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "KCS12_", date_time, "_", filename])
###########################################################################################################

cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath=modelpath)

start_time = time.time()
hist = model.fit([x1_train, x2_train], [y_train], epochs=100, batch_size=256, verbose=1, callbacks=[es, cp], validation_split=0.2)
end_time = time.time() - start_time

# model.save('./samsung/_save/ModelCheckPoint/KCS12_model_save.hdf5')

# model = load_model('./samsung/_save/ModelCheckPoint/KCS12_model_save.hdf5')

# #4. 평가, 예측
results = model.evaluate([x1_test, x2_test], y_test)
y_predict = model.predict([x1_predict, x2_predict])
# r2 = r2_score(y_test, y_predict)
print('걸린 시간 : ', end_time)
print('loss: ',results[0])
print('mse: ',results[1])
print('2021년 9월 1일 주택가격지수 : ', y_predict[-1])
# print("r2스코어 :", r2)


#5. plt 시각화
# plt.figure(figsize=(9,5))

# #1
# plt.subplot(2,1,1)
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# #2
# plt.subplot(2,1,2)
# plt.plot(hist.history['mae'])
# plt.plot(hist.history['val_mae'])
# plt.grid()
# plt.title('mae')
# plt.ylabel('mae')
# plt.xlabel('epoch')
# plt.legend(['mae', 'val_mae'])

# plt.show()



# 예측 결과 시각화
# a_axis = np.arange(0, len(y_train))
# b_axis = np.arange(len(y_train), len(y_train) + len(y_predict))

# plt.figure(figsize=(10,6))
# plt.plot(a_axis, y_train, 'o-')
# plt.plot(b_axis, y_predict, 'o-', color='red', label='Predicted')
# plt.plot(b_axis, y_test, 'o-', color='green', alpha=0.2, label='Actual')
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 9))
# plt.plot(y_test, label='actual')
# plt.plot(y_predict, label='prediction')
# plt.legend()
# plt.show()


# 걸린 시간 :  29.82509446144104
# loss:  32.5546760559082
# mse:  4.473262786865234
# 2021년 9월 1일 주택가격지수 :  [104.883484]

# 걸린 시간 :  56.56564903259277
# loss:  30.047361373901367
# mse:  4.222527503967285
# 2021년 9월 1일 주택가격지수 :  [103.24317]

# 걸린 시간 :  87.74114394187927
# loss:  14.826424598693848
# mse:  2.7652475833892822
# 2021년 9월 1일 주택가격지수 :  [101.60072]

# 걸린 시간 :  73.66404604911804
# loss:  18.088268280029297
# mse:  3.1695456504821777
# 2021년 9월 1일 주택가격지수 :  [99.84182]

# 걸린 시간 :  51.77107238769531
# loss:  29.643537521362305
# mse:  4.144094467163086
# 2021년 9월 1일 주택가격지수 :  [102.47518]


# 걸린 시간 :  55.77410054206848
# loss:  37.770774841308594
# mse:  5.20835542678833
# 2021년 9월 1일 주택가격지수 :  [97.880165]

# 걸린 시간 :  71.97701334953308
# loss:  24.510316848754883
# mse:  3.6051857471466064
# 2021년 9월 1일 주택가격지수 :  [100.42258]


# 걸린 시간 :  248.08157467842102
# loss:  9.901674270629883
# mse:  2.146397352218628
# 2021년 9월 1일 주택가격지수 :  [99.24788]
