from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder


#이미지가  32, 32, 3 칼라

# 완성하시오

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)


# 전처리

x_train = x_train.reshape(50000, 32, 32, 3)/255.  
x_test = x_test.reshape(10000, 32, 32, 3)/255.

en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 10)
print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 10)

# scaler = MinMaxScaler()
# scaler.fit(x_train) # 훈련
# x_train = scaler.transform(x_train) # 변환
# x_test = scaler.transform(x_test)

# print(x_train.shape)
# print(x_test.shape)

# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]


#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(16, (2,2), padding='same', activation='relu'))    
model.add(Conv2D(32, (2,2), padding='same', activation='relu')) 
model.add(MaxPool2D())
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))  
model.add(Flatten())    
model.add(Dense(128, activation='relu'))  
model.add(Dense(64,  activation='relu'))  
model.add(Dense(32, activation='relu'))  
model.add(Dense(16, activation='relu'))  
model.add(Dense(10, activation='softmax'))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=512, validation_split=0.2, callbacks=[es])

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
loss :  3.836859703063965
accuracy :  0.6022999882698059

loss :  4.0142974853515625
accuracy :  0.6718999743461609

loss :  4.020190238952637
accuracy :  0.7013000249862671
'''
