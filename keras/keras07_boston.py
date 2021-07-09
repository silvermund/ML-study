from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

# 완료하시오!!!

datasets = load_boston()

x = datasets.data
y = datasets.target

print(x.shape)
# print(x.shape)
# print(datasets.feature_names)
# print(datasets.DESCR)

'''
model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=100, batch_size=1)

loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('x_predict의 예측값 : ', y_predict)

r2 = r2_score(y, y_predict)

print("r2스코어 :", r2)
'''

