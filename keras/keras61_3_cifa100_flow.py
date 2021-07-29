# 훈련데이터를 10만개로 증폭할 것!!
# 완료 후 기존 모델과 비교
# save_dir도 temp에 넣을 것
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 본 뒤 삭제

# flow to 100,000 
# make model and compare with banila
# save_dir -> temp and delete

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Embedding, LSTM, Flatten, Dropout, Bidirectional, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()


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

augment_size = 50000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0])     #50000
print(randidx)              # [24106 22472 20189 ... 10321 35247 47771]
print(randidx.shape)        #(50000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape) #(50000, 32, 32, 3)


x_augmented = x_augmented.reshape(x_augmented.shape[0], 32, 32, 3)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)


x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                batch_size=augment_size, shuffle=False,
                                #save_to_dir='../temp/'  # 이번파일은 요놈이 주인공!!
                                ).next()[0]


# print(x_augmented.shape) #(40000, 28, 28, 1)


x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)

x_train = x_train.reshape(100000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)
y_train = y_train.reshape(100000, 1)
y_test = y_test.reshape(10000, 1)


en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()

print(x_test.shape, y_test.shape)

# 2. 모델

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(20, (2,2), activation='relu'))    
model.add(Conv2D(30, (2,2), padding='valid')) 
model.add(MaxPool2D())
model.add(Conv2D(15, (2,2)))  
model.add(Flatten())    
model.add(Dense(64, activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(100, activation='sigmoid'))

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


# after flow
# 걸린 시간 :  208.15932297706604
# acc :  0.36793750524520874
# val_acc :  0.15115000307559967

