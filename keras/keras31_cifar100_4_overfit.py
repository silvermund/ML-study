# overfit을 극복하자!!!
# 1. 전체 훈련 데이터가 많이 많이!!
# 2. normalization 
# 3. dropout

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
import time


#이미지가  32, 32, 3 칼라

# 완성하시오

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)


# 전처리

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

# print(x_train.shape, x_test.shape) #(50000, 3072) (10000, 3072)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 변환
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32 , 32 , 3)  
x_test = x_test.reshape(10000, 32 , 32 , 3)

# print(x_train.shape)
# print(x_test.shape)


# en = OneHotEncoder()
# y_train = en.fit_transform(y_train).toarray()
# y_test = en.fit_transform(y_test).toarray()

# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 10)
# print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 10)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='valid', input_shape=(32, 32, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))   
model.add(MaxPool2D())

model.add(Conv2D(128, (2,2), padding='valid', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
model.add(MaxPool2D())

model.add(Conv2D(64, (2,2), padding='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))  
model.add(MaxPool2D())

model.add(Flatten())    
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))  
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu')) 
model.add(Dense(100, activation='softmax'))

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=1024, validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=64)
print("==============평가, 예측==============")
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# print("=================예측===============")
# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)

#5. plt 시각화
plt.figure(figsize=(9,5))

#1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

#2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

'''
loss :  3.580064535140991
accuracy :  0.1623000055551529

loss :  4.3922224044799805
accuracy :  0.19449999928474426

loss :  9.623822212219238
accuracy :  0.2969000041484833

걸린 시간 :  1056.0345032215118
loss :  11.645252227783203
accuracy :  0.2782000005245209

standardscaler
걸린 시간 :  511.63596987724304
loss :  11.798748016357422
accuracy :  0.27309998869895935

걸린 시간 :  265.52802634239197
loss :  13.699191093444824
accuracy :  0.23800000548362732

걸린 시간 :  540.6912522315979
loss :  13.195276260375977
accuracy :  0.23970000445842743

걸린 시간 :  265.52802634239197
loss :  13.699191093444824
accuracy :  0.23800000548362732

걸린 시간 :  1004.4971594810486
loss :  6.938262462615967
accuracy :  0.301800012588501

걸린 시간 :  506.8670868873596
loss :  6.971688270568848
accuracy :  0.35249999165534973

dropout 훈련
걸린 시간 :  1096.8704738616943
loss :  2.2400619983673096
accuracy :  0.4291999936103821


걸린 시간 :  1022.8968975543976
loss :  2.0122263431549072
accuracy :  0.46810001134872437
'''

