from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv1D, Flatten, MaxPool1D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
import time


datasets = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)

# ./  : 현재폴더
# ../ : 상위폴더

# print(datasets)
# print(datasets.shape) # (4898, 12)

# 11개까지 x, 마지막 1개 quality가 y

# print(datasets.info())
# print(datasets.describe())

# 다중분류
# 모델링하고
# 0.8 이상 완성!!!


#1. 데이터
np = datasets.values


x = np[:,:11]
y = np[:,11:]

print(x)
print('-----------------------------------------')
print(y)

#1. 판다스 -> 넘파이
#2. x와 y를 분리
#3. sklearn의 onehot??? 사용할 것 
#4. y의 라벨을 확인 np.unique(y)
#5. y의 shape 확인 (4898, ) -> (4898,7)


print(x.shape, y.shape) # (4898, 11) (4898, 1)

print(y) # y가 0,1,2
# print(np.unique(y))

ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)

# y = to_categorical(y)

print(y[:5])
print(y.shape) # (4898, 7)

print(type(x), type(y))


# 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

scaler = StandardScaler()
scaler.fit(x_train) # 훈련
x_train = scaler.transform(x_train) # 변환
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape) #(3428, 11, 1)
print(x_test.shape) #(1470, 11, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, padding='same', input_shape=(11,1)))
model.add(Dropout(0.2))
model.add(Conv1D(16, 2, padding='same', activation='relu'))   
model.add(MaxPool1D())

model.add(Conv1D(64, 2, padding='same', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', activation='relu')) 
model.add(MaxPool1D())

model.add(Conv1D(256, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(256, 2, padding='same', activation='relu'))  
# model.add(MaxPool1D())

# model.add(Flatten())    
# model.add(Dense(128, activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))  
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu')) 
model.add(GlobalAveragePooling1D())
model.add(Dense(7, activation="softmax"))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time



# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])


# plt.plot(hist.history['loss'])  # x:epoch, / y:hist.history['loss']
# plt.plot(hist.history['val_loss'])

# plt.title("loss, val_loss")
# plt.xlabel('epoch')
# plt.ylabel('loss, val_loss')
# plt.legend(['train loss','val loss'])
# plt.show()

'''
loss :  1.8325819969177246
accuracy :  0.5442177057266235

원핫인코더
loss :  1.6113581657409668
accuracy :  0.5673469305038452

StandardScaler
loss :  2.4856913089752197
accuracy :  0.5965986251831055

PowerTransformer
loss :  2.5612287521362305
accuracy :  0.6000000238418579

cnn
time :  44.05618762969971
loss :  1.4128811359405518
acc :  0.6319727897644043
'''
