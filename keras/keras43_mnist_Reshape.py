from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv2D, Flatten, MaxPooling2D, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# 전처리

# x_train = x_train.reshape(60000, 784)  
# x_test = x_test.reshape(10000, 784)
y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)

en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

# x_train, x_test = x_train/255.0, x_test/255.0

# scaler = MinMaxScaler()
# scaler.fit(x_train) # 훈련
# x_train = scaler.transform(x_train) # 변환
# x_test = scaler.transform(x_test)

print(x_train.shape)
print(x_test.shape)

# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

model = Sequential()
# model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', input_shape=(28, 28, 1)))
# model.add(Conv2D(20, (2,2), activation='relu'))    
# model.add(Conv2D(30, (2,2), padding='valid')) 
# model.add(MaxPool2D())
# model.add(Conv2D(15, (2,2)))  
# model.add(Flatten())    
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(10, activation='sigmoid'))

model.add(Dense(units=10, activation='relu', input_shape=(28,28)))
model.add(Flatten())     # (None, 280)
model.add(Dense(784))    # (None, 784)
model.add(Reshape((28, 28, 1)))   # (None, 28, 28, 1)
model.add(Conv2D(64, (2,2)))
model.add(Conv2D(64, (2,2)))
model.add(Conv2D(64, (2,2)))
model.add(MaxPooling2D())
model.add(Flatten())  
model.add(Dense(16))
model.add(Dense(8))
model.add(Reshape(4,2,1))
model.add(Dense(10, activation='softmax'))

model.summary()
'''
#2. 모델 구성

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.2, callbacks=[es])

# print(hist.history.keys())
# print("================================")
# print(hist.history['loss'])
# print("================================")
# print(hist.history['val_loss'])
# print("================================")



#4. 평가, 예측
print("==============평가, 예측=============")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# print("=================예측===============")
# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)
'''
loss :  0.1420455276966095
accuracy :  0.9817000031471252

dnn
loss :  0.18461468815803528
accuracy :  0.9465000033378601
'''
'''