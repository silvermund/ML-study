from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate
import numpy as np
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

datasets = load_diabetes()

#1. 데이터
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(442,10) (442,)

print(datasets.feature_names) #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))

# 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=5)

scaler = PowerTransformer()
scaler.fit(x_train) # 훈련
x_train = scaler.transform(x_train) # 변환
x_test = scaler.transform(x_test)


#2. 모델구성
input1 = Input(shape=(10,))
dense1 = Dense(64)(input1)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
dense5 = Dense(32, activation='relu')(dense4)
dense6 = Dense(32, activation='relu')(dense5)
dense7 = Dense(8, activation='relu')(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)
model.summary()


# model = Sequential()
# model.add(Dense(2048, input_dim=10)) 
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, shuffle=True)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# print('x_predict의 예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict)

print("r2스코어 :", r2)

'''
MaxAbsScaler 선택
loss :  5734.16357421875
r2스코어 : 0.09063018175482285

RobustScaler 선택
loss :  4946.02197265625
r2스코어 : 0.2156200288326543

QuantileTransformer 선택
loss :  2891.48095703125
r2스코어 : 0.5414456935238593

PowerTransformer 선택
loss :  4968.982421875
r2스코어 : 0.21197883181389277

'''