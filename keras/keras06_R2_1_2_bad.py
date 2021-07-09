#1. R2를 음수가 아닌 0.5 이하로 만들어라.
#2. 데이터 건들지 마
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4. batch_size = 1
#5. epochs는 100 이상
#6. 히든레이어의 노드는 10개 이상 1000개 이하
#7. trian 70%


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터

x = np.array(range(100))
y = np.array(range(1, 101))

# x_train = x[:70]
# y_train = y[:70]
# x_test = x[-30:]
# y_test = y[70:]

# print(x_train.shape, y_train.shape) # (70,) (70,)
# print(y_train.shape, y_test.shape)  # (30,) (30,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

print()
#2. 모델 구성
model = Sequential()
model.add(Dense(6, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('101의 예측값 : ', y_predict)

# y_predict = model.predict(x)

# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()

r2 = r2_score(y_test, y_predict)

print("r2스코어 :", r2)
