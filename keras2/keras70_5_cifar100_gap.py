# 실습

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
import time

from tensorflow.python.keras.layers.convolutional import Conv1D


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 변환
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32 , 32 , 3)  
x_test = x_test.reshape(10000, 32 , 32 , 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 10)
# print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 10)



#2. 모델 구성
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
# model = VGG16()
# model = VGG19()

vgg16.trainable=True   # 가중치를 동결한다, 훈련을 동결한다


model = Sequential()

model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(2048, activation='relu'))
# model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))


# model.trainable=False  # 가중치를 동결한다, 훈련을 동결한다


model.summary()

print(len(model.weights))             # 26 -> 30
print(len(model.trainable_weights))   # 0 -> 4



#3. 컴파일, 훈련
# optimizer = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=64)
print("==============평가, 예측==============")
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# vgg16.trainable=False
# ==============평가, 예측==============
# 걸린 시간 :  63.857659101486206
# loss :  3.1040680408477783
# accuracy :  0.4101000130176544

# vgg16.trainable=True
# ==============평가, 예측==============
# 걸린 시간 :  158.4190924167633
# loss :  4.007357597351074
# accuracy :  0.35370001196861267