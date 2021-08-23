# 실습, 과제
# keras61_5 남자 여자 데이터에 노이즈를 넣어서
# 기미 주근깨 여드름 제거하시오!!!

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Embedding, LSTM, Flatten, Dropout, Bidirectional, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import time

# 1. 데이터

x_train = np.load('./_save/_npy/k59_5_train_x.npy')
y_train = np.load('./_save/_npy/k59_5_train_y.npy')
x_test = np.load('./_save/_npy/k59_5_test_x.npy')
y_test = np.load('./_save/_npy/k59_5_test_y.npy')
x_selfy = np.load('./_save/_npy/k59_5_selfy_x.npy')
y_selfy = np.load('./_save/_npy/k59_5_selfy_y.npy')

print(x_train.shape, y_train.shape) #(2482, 150, 150, 3) (2482,)
print(x_test.shape, y_test.shape)   #(827, 150, 150, 3) (827,)
# print(x_selfy.shape, y_selfy.shape) # (1, 150, 150, 3) (1,)

# print(x_selfy, y_selfy)

x_train1 = x_train.reshape(2482, 150, 150, 3).astype('float')/255.
x_train2 = x_train.reshape(2482, 150*150*3).astype('float')/255.
x_test = x_test.reshape(827, 150, 150, 3).astype('float')/255.

x_train_noised = x_train1 + np.random.normal(0, 0.1, size=x_train1.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


# 2. 모델

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters = 128, kernel_size=(3,3),  activation= 'relu'))
model.add(Conv2D(filters = 128, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))




# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32, #160/5=32
#                             validation_data=xy_test,  
#                             validation_steps=4)  

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, validation_split=0.2, 
                steps_per_epoch=32, validation_steps=4, callbacks=[es])
end_time = time.time() - start_time





# 4. 평가, 예측
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_selfy)
res = (1-y_predict) * 100

print('걸린 시간 : ', end_time)
print('loss: ',results[0])
print('acc: ',results[1])
print('남자일 확률 : ', res, '%')
print('남0, 여1 :', np.argmax(y_predict[0]))

print("acc : ", acc[-1])
print("val_acc : ", val_acc[-1])


