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

print(x_test)
#[ 8 93  4  5 52 41  0 73 88 68 25 18 26 29 66 50 80 45 38 58 49 85 94 87
#  15  3 14 33 23 24]
print(y_test)
#[ 9 94  5  6 53 42  1 74 89 69 26 19 27 30 67 51 81 46 39 59 50 86 95 88
#  16  4 15 34 24 25]

#2. 모델 구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=120, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([y_test])
print('100의 예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict)

print("r2스코어 :", r2)


#5. 시각화
# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()


'''
loss :  0.001323983888141811
100의 예측값 :  [[10.052702 ]
 [94.99945  ]
 [ 6.0552063]
 [ 7.0545793]
 [54.025127 ]
 [43.03202  ]
 [ 2.0577111]
 [75.01198  ]
 [90.00259  ]
 [70.01512  ]
 [27.042055 ]
 [20.046434 ]
 [28.041424 ]
 [31.039547 ]
 [68.016365 ]
 [52.026386 ]
 [82.00761  ]
 [47.02951  ]
 [40.0339   ]
 [60.021374 ]
 [51.027008 ]
 [87.00446  ]
 [95.99883  ]
 [89.00322  ]
 [17.048317 ]
 [ 5.055832 ]
 [16.048939 ]
 [35.03704  ]
 [25.043303 ]
 [26.042679 ]]
r2스코어 : 0.9987846660979255
'''