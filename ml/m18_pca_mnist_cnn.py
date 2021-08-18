# 실습
# mnist 데이터를 pca를 통해 cnn으로 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv1D, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA(n_components=196)
x = pca.fit_transform(x)


#print(x.shape) #(70000, 196)

#print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)


# x = np.append(x_train, x_test, axis=0)
# print(x.shape) #(70000, 28, 28)
# x = x.reshape(70000, 14, 14)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

# print(x_train.shape, x_test.shape) #(56000, 196) (14000, 196)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 변환
x_test = scaler.transform(x_test)

x_train = x_train.reshape(56000, 196, 1)
x_test = x_test.reshape(14000, 196, 1)

y_train = y_train.reshape(56000,1)
y_test = y_test.reshape(14000,1)

en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()

# print(y_train.shape) #(56000, 10)
# print(y_test.shape)  #(14000, 10)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))

# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

# print(np.argmax(cumsum >= 0.95)+1) #154

# plt.plot(cumsum)
# plt.grid()
# plt.show()

#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(14*14, 1)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10))

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=64)
print("==============평가, 예측==============")
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# dnn 713
# 걸린 시간 :  62.68300819396973
# loss :  0.7265920639038086
# accuracy :  0.9369999766349792

# dnn 196
# 걸린 시간 :  33.90214443206787
# loss :  0.29685407876968384
# accuracy :  0.9551428556442261

# cnn
# 걸린 시간 :  41.45934057235718
# loss :  6.2595438957214355
# accuracy :  0.18528571724891663
