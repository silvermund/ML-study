# 실습

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.applications import DenseNet121
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


x_train=tf.image.resize(x_train,[75,75])
x_test=tf.image.resize(x_test,[75,75])

#2. 모델 구성
denseNet121 = DenseNet121(weights='imagenet', include_top=False, input_shape=(75,75,3))
# model = VGG16()
# model = VGG19()


denseNet121.trainable=False   # 가중치를 동결한다, 훈련을 동결한다

model = Sequential()
model.add(denseNet121)
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(10))

model.trainable=False # 가중치를 동결한다, 훈련을 동결한다

model.summary()

print(len(model.weights))             # 26 -> 30
print(len(model.trainable_weights))   # 0 -> 4



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=1024, validation_split=0.25) #,callbacks=[es])
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
# 걸린 시간 :  440.41687536239624
# loss :  9.619283676147461
# accuracy :  0.10000000149011612

# vgg19.trainable=True
# model.trainable=False
# ==============평가, 예측==============
# 걸린 시간 :  67.48802828788757
# loss :  6.619750022888184
# accuracy :  0.10540000349283218

# vgg19.trainable=False
# model.trainable=True
# ==============평가, 예측==============
# 걸린 시간 :  472.6768114566803
# loss :  8.25407600402832
# accuracy :  0.10010000318288803


# vgg19.trainable=False
# model.trainable=False
# ==============평가, 예측==============
# 걸린 시간 :  69.52321124076843
# loss :  8.17580509185791
# accuracy :  0.08900000154972076