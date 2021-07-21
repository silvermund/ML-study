from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv1D, Conv2D, Flatten, MaxPool2D, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
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

x_train = x_train.reshape(50000, 32*32, 3)
x_test = x_test.reshape(10000, 32*32, 3)

#2. 모델 구성
model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(32 * 32 * 3,)))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()

# input1 = Input(shape=(32*32, 3))
# xx = LSTM(4, activation='relu')(input1)
# xx = Dense(2, activation='relu')(xx)
# output1 = Dense(100, activation='softmax')(xx)

# model = Model(inputs=input1, outputs=output1)

model.add(Conv1D(64, 2, input_shape=(32*32, 3)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(100))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=2048, validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("==============평가, 예측==============")
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# print("=================예측===============")
# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)
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
loss :  3.836859703063965
accuracy :  0.6022999882698059

loss :  4.0142974853515625
accuracy :  0.6718999743461609

loss :  4.020190238952637
accuracy :  0.7013000249862671

걸린 시간 :  131.46661972999573
loss :  3.6137051582336426
accuracy :  0.4287000000476837

lstm
걸린 시간 :  174.34751510620117
loss :  nan
accuracy :  0.009999999776482582

conv1d
걸린 시간 :  53.654568910598755
loss :  8.078387260437012
accuracy :  0.009999999776482582
'''
