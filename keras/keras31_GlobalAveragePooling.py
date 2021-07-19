from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D
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

# model.add(Flatten())    
# model.add(Dense(128, activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))  
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu')) 
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=512, validation_split=0.25, callbacks=[es])
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
dropout 훈련
걸린 시간 :  1096.8704738616943
loss :  2.2400619983673096
accuracy :  0.4291999936103821

걸린 시간 :  1022.8968975543976
loss :  2.0122263431549072
accuracy :  0.46810001134872437

global average pooling
걸린 시간 :  1014.0808243751526
loss :  2.082434892654419
accuracy :  0.46549999713897705

gap epoch 512
걸린 시간 :  656.5321109294891
loss :  2.0789260864257812
accuracy :  0.476500004529953

gap epoch 128
'''

