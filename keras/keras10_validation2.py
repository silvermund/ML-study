from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

## 잘라서 만들어라!!!

x_train = x[:8] # 훈련, 공부하는 거
y_train = y[:8]
x_test = x[8:11]
y_test = y[8:11]
x_val = x[11:]
y_val = y[11:]

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([11])
print('11의 예측값 : ', result)

# y_predict = model.predict(x)

# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()
