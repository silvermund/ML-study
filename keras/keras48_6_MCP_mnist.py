from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv1D, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# 전처리

x_train = x_train.reshape(60000, 28 * 28)  
x_test = x_test.reshape(10000, 28 * 28)
y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 변환
x_test = scaler.transform(x_test)

en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()
# print(x_train.shape) #(60000, 784)

# print(x_test.shape) #(10000, 784)

# print(y_train.shape) #(60000, 10)
# print(y_test.shape) #(10000, 10)

x_train = x_train.reshape(60000, 784, 1)
x_test = x_test.reshape(10000, 784, 1)

print(y_train.shape) #(60000, 10)
print(y_test.shape) #(10000, 10)


#2. 모델 구성
model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(28 * 28,)))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(512, activation='relu'))

# model.add(Dense(10, activation='softmax'))

# model.summary()

# input1 = Input(shape=(28*28, 1))
# xx = LSTM(4, activation='relu')(input1)
# xx = Dense(2, activation='relu')(xx)
# output1 = Dense(10, activation='softmax')(xx)

# model = Model(inputs=input1, outputs=output1)

model.add(Conv1D(64, 2, input_shape=(28*28, 1)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10))




#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath='./_save/ModelCheckPoint/keras48_6_MCP.hdf5')


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=2048, validation_split=0.2, callbacks=[es, cp])
end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_6_model_save.hdf5')

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
loss :  0.1420455276966095
accuracy :  0.9817000031471252

dnn
걸린 시간 :  48.73673367500305
loss :  0.2552763521671295
accuracy :  0.9749000072479248

lstm
걸린 시간 :  960.5627167224884
loss :  2.3012163639068604
accuracy :  0.11349999904632568

conv1d
걸린 시간 :  38.396968126297
loss :  nan
accuracy :  0.09799999743700027

model
걸린 시간 :  43.27663254737854
loss :  7.722182750701904
accuracy :  0.10339999943971634

MCP
걸린 시간 :  34.823872804641724
loss :  7.937629699707031
accuracy :  0.12549999356269836
'''
