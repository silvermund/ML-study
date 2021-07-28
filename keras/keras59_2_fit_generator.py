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
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)
# Found 120 images belonging to 2 classes.


# # print(xy_train)
# # # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000251165B8550>
# # # print(xy_train[0])
# # # y가 5개 = batch_size
# # # print(xy_train[0][0])        # x값
# # # print(xy_train[0][1])        # y값
# # # print(xy_train[0][2])      # 없음
# # print(xy_train[0][0].shape, xy_train[0][1].shape)       # (5, 150, 150, 3) (5,)
# #                                                     #배치사이즈
# # 160 / 5 = 32 => [0]~[31]
# #[31][0] = 0, [31][0] = 1
# # print(xy_train[32][1]) => 없음

# print(type(xy_train))           #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))        #<class 'tuple'>
# print(type(xy_train[0][0]))     #<class 'numpy.ndarray'>
# print(type(xy_train[0][1]))     #<class 'numpy.ndarray'>

# 2. 모델

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32, #160/5=32
                            validation_data=xy_test,  
                            validation_steps=4)  


# 4. 평가, 예측
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위에거로 시각화 할 것

print("acc : ", acc[-1])
print("val_acc : ", val_acc[-1])

# acc :  0.606249988079071
# val_acc :  0.5916666388511658