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

augment_size=50
x_data = train_datagen.flow(
            np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),  # x값
            np.zeros(augment_size),   # y값
            batch_size=augment_size,
            shuffle=False
).next()                                # iterator 방식으로 반환!!

print(type(x_data)) #<class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
                    # ->   <class 'tuple'>
print(type(x_data[0])) #<class 'tuple'>
                       # -> <class 'numpy.ndarray'>
# print(type(x_data[0][0])) #<class 'numpy.ndarray'>
print(x_data[0][0].shape) #(100, 28, 28, 1) -> x값
                          #(28, 28, 1)
print(x_data[0].shape)    
                        #(100, 28, 28, 1)
print(x_data[1].shape) #(100,)           -> y값


plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()