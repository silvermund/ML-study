from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = [1,2,3,4,5]
y = [1,2,3,4,5]
x_pred = [6]


print(x.shape)

# 완성한 뒤, 출력결과스샷 후 메일로 보내기(loss, pred)

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=100, batch_size=10)

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('6의 예측값 : ', result)