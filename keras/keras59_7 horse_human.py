# import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(
#     rescale=1./255, 
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=1.2,
#     shear_range=0.7,
#     fill_mode='nearest'
# )
# test_datagen = ImageDataGenerator(rescale=1./255)

# train = train_datagen.flow_from_directory(
#     '../_data/horse-or-human',
#     target_size=(150, 150),
#     batch_size=1100,
#     class_mode='categorical',
#     classes=['horses','humans'],
#     shuffle=True
# )


# test = test_datagen.flow_from_directory(
#     '../_data/horse-or-human',
#     target_size=(150, 150),
#     batch_size=1100,
#     class_mode='categorical',
#     classes=['horses','humans'],
#     shuffle=True
# )



# print(xy_train[0])
# y가 5개 = batch_size
# print(train[0][0])        # x값
# print(train[0][1])        # y값
# print(xy_train[0][2])      # 없음

# print(train[0][0].shape, train[0][1].shape)     # (1027, 150, 150, 3) (1027, 2)
# print(test[0][0].shape, test[0][1].shape)       # (1027, 150, 150, 3) (1027, 2)



# 160 / 5 = 32 => [0]~[31]
# [31][0] = 0, [31][0] = 1
# print(xy_train[32][1]) => 없음

# print(type(xy_train))           #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))        #<class 'tuple'>
# print(type(xy_train[0][0]))     #<class 'numpy.ndarray'>
# print(type(xy_train[0][1]))     #<class 'numpy.ndarray'>


# np.save('./_save/_npy/k59_7_train_x.npy', arr=train[0][0])
# np.save('./_save/_npy/k59_7_train_y.npy', arr=train[0][1])
# np.save('./_save/_npy/k59_7_test_x.npy', arr=test[0][0])
# np.save('./_save/_npy/k59_7_test_y.npy', arr=test[0][1])


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Embedding, LSTM, Flatten, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import time

# 1. 데이터

x_train = np.load('./_save/_npy/k59_7_train_x.npy')
y_train = np.load('./_save/_npy/k59_7_train_y.npy')
x_test = np.load('./_save/_npy/k59_7_test_x.npy')
y_test = np.load('./_save/_npy/k59_7_test_y.npy')

# print(x_train.shape, y_train.shape) #(1027, 150, 150, 3) (1027, 2)
# print(x_test.shape, y_test.shape)   #(1027, 150, 150, 3) (1027, 2)


# 2. 모델

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size=(2,2), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters = 16, kernel_size=(2,2),  activation= 'relu'))
# model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 64, kernel_size=(2,2),  activation= 'relu'))
model.add(Conv2D(filters = 64, kernel_size=(2,2),  activation= 'relu'))
# model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(2, activation= 'softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32, #160/5=32
#                             validation_data=xy_test,  
#                             validation_steps=4)  

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가, 예측
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
results = model.evaluate(x_test, y_test)
# y_predict = model.predict(x_test)

print('걸린 시간 : ', end_time)
print('loss: ',results[0])
print('acc: ',results[1])


# print("acc : ", acc[-1])
# print("val_acc : ", val_acc[-1])


# loss:  0.419383704662323
# acc:  0.840311586856842