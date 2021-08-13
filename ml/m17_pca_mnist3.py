from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv2D, Flatten, MaxPool2D
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

pca = PCA(n_components=713)
x = pca.fit_transform(x)


#print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

# x = np.append(x_train, x_test, axis=0)
# print(x.shape) #(70000, 28, 28)
# x = x.reshape(70000, 28*28)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 변환
x_test = scaler.transform(x_test)

en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()

# 실습
# pca를 통해 1.0 이상인게 몇개?





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
model.add(Dense(128, activation='relu', input_shape=(713,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Dense(10, activation='softmax'))

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

# 걸린 시간 :  62.68300819396973
# loss :  0.7265920639038086
# accuracy :  0.9369999766349792
