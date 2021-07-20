# 결과값이 80 나오도록 튜닝

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9,], [8,9,10], [9,10,11],
            [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape, y.shape) #(13, 3) (13,)



# x = x.reshape(13, 3, 1)  #(batch_size, timesteps, feature)
# x_predict = x_predict.reshape(1,3,1)

x = x.reshape(x.shape[0], x.shape[1], 1)
x_predict = x_predict.reshape(1, x_predict.shape[0], 1)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(32, activation='relu', input_shape=(3, 1))) # 행무시
# model.add(LSTM(32, activation='relu', input_length=3, input_dim=1))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

#4. 평가, 예측
results = model.predict(x_predict)
print(results)

# (input + bias) * output + output * output
# = (input + bias + output) * output
# 120 = 10 (12=10+1+1)
# 288 = 16 (18=16+1+1)

'''
[[82.22839]]
'''