# 06_R2_2를 카피
# 함수형으로 리폼하시오.
# 서머리로 확인

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_prdict=[6]


input1 = Input(shape=(1,))
dense1 = Dense(6)(input1)
dense2 = Dense(5)(dense1)
dense3 = Dense(4)(dense2)
dense4 = Dense(3)(dense3)
dense5 = Dense(2)(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)
model.summary()

# model = Sequential()
# model.add(Dense(6, input_dim=1))
# model.add(Dense(5))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(2))
# model.add(Dense(1))

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