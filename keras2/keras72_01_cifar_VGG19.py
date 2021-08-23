# 실습

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
import time

from tensorflow.python.keras import activations


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train = x_train.reshape(50000, 32 * 32 * 3)
# x_test = x_test.reshape(10000, 32 * 32 * 3)


en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 10)
print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 10)



#2. 모델 구성
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))
# model = VGG16()
# model = VGG19()

vgg19.trainable=False  # 가중치를 동결한다, 훈련을 동결한다

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

model.trainable=False  # 가중치를 동결한다, 훈련을 동결한다

model.summary()

print(len(model.weights))             # 26 -> 30
print(len(model.trainable_weights))   # 0 -> 4



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size=2048, validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=64)
print("==============평가, 예측==============")
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# ==============평가, 예측==============
# 걸린 시간 :  553.2899568080902
# loss :  nan
# accuracy :  0.10000000149011612


# vgg19 t/t
# ==============평가, 예측==============
# 걸린 시간 :  99.05072283744812
# loss :  2.302706718444824
# accuracy :  0.10000000149011612

# vgg19 t/f
# ==============평가, 예측==============
# 걸린 시간 :  29.8152916431427
# loss :  41.52933120727539
# accuracy :  0.09719999879598618

# vgg19 f/t
# ==============평가, 예측==============
# 걸린 시간 :  252.52189588546753
# loss :  1.1134307384490967
# accuracy :  0.715399980545044

# vgg19 f/f
# ==============평가, 예측==============
# 걸린 시간 :  29.099140167236328
# loss :  46.398048400878906
# accuracy :  0.10679999738931656