# 실습

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
import time
import tensorflow as tf




(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train = x_train.reshape(50000, 32 * 32 * 3)
# x_test = x_test.reshape(10000, 32 * 32 * 3)


en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 10)
print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 10)


x_train=tf.image.resize(x_train,[71,71])
x_test=tf.image.resize(x_test,[71,71])

#2. 모델 구성
resNet101 = ResNet101(weights='imagenet', include_top=False, input_shape=(71,71,3))
# model = VGG16()
# model = VGG19()


resNet101.trainable=False   # 가중치를 동결한다, 훈련을 동결한다

model = Sequential()
model.add(resNet101)
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(10))

model.trainable=False  # 가중치를 동결한다, 훈련을 동결한다

model.summary()

print(len(model.weights))             # 26 -> 30
print(len(model.trainable_weights))   # 0 -> 4



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=512, validation_split=0.25) #,callbacks=[es])
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=64)
print("==============평가, 예측==============")
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


# vgg19.trainable=True
# model.trainable=True
# ==============평가, 예측==============
# 걸린 시간 :  338.4012050628662
# loss :  8.499072074890137
# accuracy :  0.14630000293254852

# vgg19.trainable=True
# model.trainable=False
# ==============평가, 예측==============
# 걸린 시간 :  126.96786046028137
# loss :  7.56681489944458
# accuracy :  0.1251000016927719

# vgg19.trainable=False
# model.trainable=True
# ==============평가, 예측==============
# 걸린 시간 :  342.4369297027588
# loss :  7.39475154876709
# accuracy :  0.11990000307559967

# vgg19.trainable=False
# model.trainable=False
# ==============평가, 예측==============
# 걸린 시간 :  153.63235592842102
# loss :  7.026603698730469
# accuracy :  0.10100000351667404