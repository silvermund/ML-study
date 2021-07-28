# 59_3 npy를 이용해서 모델을 완성하시오

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Embedding, LSTM, Flatten
import matplotlib.pyplot as plt
import numpy as np


# 1. 데이터

x_train = np.load('./_save/_npy/k59_3_train_x.npy')
x_test = np.load('./_save/_npy/k59_3_test_x.npy')
y_train = np.load('./_save/_npy/k59_3_train_y.npy')
y_test = np.load('./_save/_npy/k59_3_test_y.npy')

# 2. 모델

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32, #160/5=32
#                             validation_data=xy_test,  
#                             validation_steps=4)  

hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)


# 4. 평가, 예측
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위에거로 시각화 할 것

print("acc : ", acc[-1])
print("val_acc : ", val_acc[-1])
