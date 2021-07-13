from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1,101), range(100), range(401,501)])
x = np.transpose(x)

print(x.shape)  #(100, 5)

y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)

print(y.shape)  #(100, 2)

# 5 ->  2

#2. 모델 구성
input1 = Input(shape=(5,))
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(2)(dense3)

model = Model(inputs=input1, outputs=output1)
model.summary()


# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))

# model.summary()

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# model.fit(x, y, epochs=100, batch_size=1)

# #4. 평가, 예측
# loss = model.evaluate(x, y)
# print('loss : ', loss)

# result = model.predict([4])
# print('4의 예측값 : ', result)
