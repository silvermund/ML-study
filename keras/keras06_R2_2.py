from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_prdict=[6]


model = Sequential()
model.add(Dense(6, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=10000, batch_size=1)

loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('x_predict의 예측값 : ', y_predict)

r2 = r2_score(y, y_predict)

print("r2스코어 :", r2)

# 과제 2
# R2를 심수안을 이겨라 !!!
# 일 밤 12시