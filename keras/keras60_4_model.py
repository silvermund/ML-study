from tensorflow.keras.datasets import fashion_mnist
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

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest'
)
# train_datagen = ImageDataGenerator(rescale=1./255)

# xy_train = train_datagen.flow_from_directory(
#     '../_data/brain/train',
#     target_size=(150, 150),
#     batch_size=5,
#     class_mode='binary',
#     shuffle=True
# )

# 1. ImageDataGenerator를 정의
# 2. 파일에서 땡겨오려면 -> flow_from_directory() // x, y가 튜플 형태로 뭉쳐있어
# 3. 데이터에서 땡겨오려면 -> flow()              // x, y가 나눠있어

augment_size=40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0])     #60000
print(randidx)              # [59679  9431   940 ... 54751 36349  4697]
print(randidx.shape)        #(40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape) #(40000, 28, 28)


x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                batch_size=augment_size, shuffle=False).next()[0]

print(x_augmented.shape) #(40000, 28, 28, 1)


x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)


# 2. 모델

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(28,28,1), activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(3,3),  activation= 'relu'))
# model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 128, kernel_size=(2,2),  activation= 'relu'))
model.add(Conv2D(filters = 128, kernel_size=(2,2),  activation= 'relu'))
# model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32, #160/5=32
#                             validation_data=xy_test,  
#                             validation_steps=4)  

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가, 예측
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
results = model.evaluate(x_train, y_train)

print('걸린 시간 : ', end_time)
print("acc : ", acc[-1])
print("val_acc : ", val_acc[-1])

# 걸린 시간 :  127.52097392082214
# acc :  0.1005999967455864
# val_acc :  0.10199999809265137


# 걸린 시간 :  160.19748783111572
# acc :  0.09982500225305557
# val_acc :  0.10199999809265137

# 모델 완성!!
# 비교 대상? loss, val_loss, acc, val_acc