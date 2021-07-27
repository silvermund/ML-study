import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape) #(4, 3) (4,)

x = x.reshape(4, 3, 1)  #(batch_size, timesteps, feature)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(10, activation='relu', input_shape=(3, 1))) # 행무시
model.add(LSTM(10, activation='relu', input_shape=(3,1), return_sequences=True))
model.add(Bidirectional(LSTM(10, activation='relu', input_length=3, input_dim=1)))
model.add(Dense(9, activation='relu'))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

#4. 평가, 예측
x_input = np.array([5,6,7]).reshape(1,3,1)

results = model.predict(x_input)
print(results)

# (input + bias) * output + output * output
# = (input + bias + output) * output
# 120 = 10 (12=10+1+1)
# 288 = 16 (18=16+1+1)



'''
[[8.3972435]]
[[8.]]
'''
