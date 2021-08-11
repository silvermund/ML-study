from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = SVC()
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과 : \n", y_predict)

results = model.evaluate(x_data, y_data)
print('model.score : ', results[1])

r2 = r2_score(y_data, y_predict)
print('r2_score : ', r2)

#tf.argmax

# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 : 
#  [[0.57918376]
#  [0.38626614]
#  [0.27475423]
#  [0.14765766]]
# 1/1 [==============================] - 0s 93ms/step - loss: 0.8171 - acc: 0.2500
# model.score :  0.8171082735061646