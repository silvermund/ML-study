from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
import time

"""
#이미지가  32, 32, 3 칼라

# 완성하시오

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #
# print(x_test.shape, y_test.shape) #


# 전처리

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

# print(x_train.shape, x_test.shape) #(50000, 3072) (10000, 3072)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) # 변환
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28 , 28 , 1)  
x_test = x_test.reshape(10000, 28 , 28 , 1)

# print(x_train.shape)
# print(x_test.shape)


# en = OneHotEncoder()
# y_train = en.fit_transform(y_train).toarray()
# y_test = en.fit_transform(y_test).toarray()


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
"""

#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='valid', input_shape=(28, 28, 1)))
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
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))  
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu')) 
model.add(Dense(10, activation='softmax'))

model.summary()

model.save('./_save/keras45_1_save_model.h5')

'''
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=64)
print("==============평가, 예측==============")
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

# print("=================예측===============")
# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)
'''
'''
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
'''
걸린 시간 :  308.08212184906006
loss :  0.03360605612397194
acc :  0.991100013256073
'''

