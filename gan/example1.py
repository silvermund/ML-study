import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras import layers, optimizers
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()


model_gen = Sequential()
model_gen.add(layers.Dense(units=3136, activation='relu'))
model_gen.add(layers.BatchNormalization())
model_gen.add(layers.Reshape(7,7,64))
model_gen.add(layers.UpSampling2D(2,2))
model_gen.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_gen.add(layers.BatchNormalization())
model_gen.add(layers.UpSampling2D(2,2))
model_gen.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_gen.add(layers.BatchNormalization())
model_gen.add(layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid'))
model_gen.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.002), metrics=['accuracy'])

model_disc = Sequential()
model_disc.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', input_shpe = (28,28,1), activation='relu'))
model_disc.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_disc.add(layers.MaxPooling2D(pool_size = (3,3), strides=2))
model_disc.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_disc.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_disc.add(layers.MaxPooling2D(pool_size = (3,3), strides=2))
model_disc.add(layers.Flatten())
model_disc.add(layers.Dense(units=1, activation='sigmoid'))
model_disc.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

#판별자로부터 생성자를 학습할 수 있도록, 생성자와 판별자를 연결 
model_comb = Sequential()
model_comb.add(model_gen)
model_comb.add(model_disc)
model_disc.trainable = False
model_comb.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

nOrig = 64
nGen = nOrig
vector_size = 10

for i in range(2000):
    y_gen = np.zeros((nGen, 28, 28))
    test_input = np.random.rand(nGen,vector_size)
    for j in range(nGen):
        o = model_gen.predict(test_input[j,:].reshape(1,10))
        o = o.reshape((28,28))
        y_gen[j,:]
    y_gen = np.expand_dims(y_gen,-1)

    nOrig = 64
    idx = np.array(range(y_train.shape[0]))
    np.random.shuffle(idx)
    idx = idx[:nOrig]
    y_orig = y_train[idx,:,:]
    y_orig = np.expand_dims(y_orig,-1)

    test_img = np.concatenate((y_gen, y_orig), 0)
    test_target = np.concatenate((np.zeros(y_gen.shape[0]), np.ones(y_gen.shape[0])),0)
    



