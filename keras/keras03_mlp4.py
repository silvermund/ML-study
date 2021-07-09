from numpy.core.fromnumeric import transpose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
x = np.array([range(10)])

x = np.transpose(x)

print(x.shape) #(10,1)

print(x)

y = np.array([[1,2,3,4,5,6,7,8,9,10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]])  #(3,10)

y = np.transpose(y)

print(y.shape) # (10,3)

print(y)



x_pred = np.array([[9]])
print(x_pred.shape) # (1,1)

print(x_pred)

#완성하시오
# #2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(3))




# #3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=100, batch_size=1)

# #4. 평가 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('예측값 : ', result)

plt.scatter(x,y[:,0])
plt.scatter(x,y[:,1])
plt.scatter(x,y[:,2])


plt.plot(x,y, color='red')
plt.show()