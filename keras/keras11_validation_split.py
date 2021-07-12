from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

## train_test_split로 만들어라!!!

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# x_train = np.array([1,2,3,4,5,6,7]) 
# y_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])         
# y_test = np.array([8,9,10])
# x_val = np.array([11,12,13])
# y_val = np.array([11,12,13])

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val))
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.3, shuffle=True)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([11])
print('11의 예측값 : ', result)

# y_predict = model.predict(x)

# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()
