from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, LSTM
from sklearn.metrics import r2_score
import numpy as np
from numpy import array


#1. 데이터

x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9,], [8,9,10], [9,10,11],
            [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70], [60,70,80], [70,80,90,], [80,90,100], [90,100,110],
            [100,110,120], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = np.array([50,60,70])
x2_predict = np.array([65,75,85])

######## 실습 : 앙상블 모델을 만드시오.

# 결과치 신경쓰지 말고 모델만 완성할 것!!!

print(x1.shape, x2.shape, y.shape) #(13, 3) (13, 3) (13,)

#2-1. 모델1
input1 = Input(shape=(3, 1))
xx = LSTM(10, activation='relu', input_length=3, input_dim=1)(input1)
xx = Dense(5, activation='relu', name='dense1')(xx)
xx = Dense(3, activation='relu', name='dense2')(xx)
xx = Dense(2, activation='relu', name='dense3')(xx)
output1 = Dense(3, name='output1')(xx)

#2-2. 모델2
input2 = Input(shape=(3, 1))
xx = LSTM(10, activation='relu', input_length=3, input_dim=1)(input2)
xx = Dense(4, activation='relu', name='dense11')(xx)
xx = Dense(4, activation='relu', name='dense12')(xx)
xx = Dense(4, activation='relu', name='dense13')(xx)
xx = Dense(4, activation='relu', name='dense14')(xx)
output2 = Dense(4, name='output2')(xx)

merge1 = Concatenate()([output1, output2])
# merge1 = concatenate([output1, output2]) #대문자로 하면 에러!!
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit([x1, x2], y, epochs=100, batch_size=8, verbose=1)

# #4. 평가, 예측
results = model.evaluate([x1, x2], y)
# print(results)

print("loss :", results[0])
print("metrics['mae'] : ", results[1])