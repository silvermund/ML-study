# 실습 1.
# men women 데이터로 모델링을 구성할 것!!!

# 실습 2.
# 본인 사진으로 predict 하시오!! d:\data 안에 본인 사진을 넣고


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
#     fill_mode='nearest',
#     validation_split=0.25
# )
# test_datagen = ImageDataGenerator(rescale=1./255)

# train = train_datagen.flow_from_directory(
#     '../_data/men_women',
#     target_size=(150, 150),
#     batch_size=3500,
#     class_mode='binary',
#     shuffle=True,
#     subset='training'
# )

# test = train_datagen.flow_from_directory(
#     '../_data/men_women',
#     target_size=(150, 150),
#     batch_size=3500,
#     class_mode='binary',
#     shuffle=True,
#     subset='validation'
# )

# selfy = test_datagen.flow_from_directory(
#     '../_data/selfy',
#     target_size=(150, 150),
#     batch_size=1,
#     class_mode='binary',
#     shuffle=True
# )

# # Found 3309 images belonging to 2 classes.
# # Found 3309 images belonging to 2 classes.


# # print(xy_train)
# # # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000251165B8550>
# # # print(xy_train[0])
# # # y가 5개 = batch_size
# # print(train[0][0])        # x값
# # print(train[0][1])        # y값
# # # print(xy_train[0][2])      # 없음

# # print(train[0][0].shape, train[0][1].shape)     # (3309, 150, 150, 3) (3309,)
# # print(test[0][0].shape, test[0][1].shape)       # (2000, 150, 150, 3) (2000,)

# print(selfy[0][0].shape, selfy[0][1].shape)       # (1, 150, 150, 3) (1,)


# # 160 / 5 = 32 => [0]~[31]
# #[31][0] = 0, [31][0] = 1
# # print(xy_train[32][1]) => 없음

# # print(type(xy_train))           #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# # print(type(xy_train[0]))        #<class 'tuple'>
# # print(type(xy_train[0][0]))     #<class 'numpy.ndarray'>
# # print(type(xy_train[0][1]))     #<class 'numpy.ndarray'>


# np.save('./_save/_npy/k59_5_train_x.npy', arr=train[0][0])
# np.save('./_save/_npy/k59_5_train_y.npy', arr=train[0][1])
# np.save('./_save/_npy/k59_5_test_x.npy', arr=test[0][0])
# np.save('./_save/_npy/k59_5_test_y.npy', arr=test[0][1])

# np.save('./_save/_npy/k59_5_selfy_x.npy', arr=selfy[0][0])
# np.save('./_save/_npy/k59_5_selfy_y.npy', arr=selfy[0][1])



from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Embedding, LSTM, Flatten, Dropout, Bidirectional, MaxPooling2D
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

print(x_train.shape, y_train.shape) #(3309, 150, 150, 3) (3309,)
print(x_test.shape, y_test.shape)   #(3309, 150, 150, 3) (3309,)
print(x_selfy.shape, y_selfy.shape) # (1, 150, 150, 3) (1,)

print(x_selfy, y_selfy)

# 2. 모델

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 128, kernel_size=(3,3),  activation= 'relu'))
model.add(Conv2D(filters = 128, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32, #160/5=32
#                             validation_data=xy_test,  
#                             validation_steps=4)  

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', verbose=1)


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
# print('남0, 여1 :', np.argmax(y_predict[0]))

# print("acc : ", acc[-1])
# print("val_acc : ", val_acc[-1])
