import numpy as np
from tensorflow.python.keras.engine import input_layer

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
# model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.001)
# loss :  0.002683038590475917 결과물 :  [[10.912115]]

# optimizer = Adagrad(lr=0.001)
# loss :  2.47513116846676e-06 결과물 :  [[10.9966345]]

# optimizer = Adagrad(lr=0.0001)
# loss :  0.0005574165843427181 결과물 :  [[10.969441]]

# optimizer = Adamax(lr=0.001)
# loss :  2.787567723316897e-07 결과물 :  [[10.999981]]

# optimizer = Adamax(lr=0.0001)
# loss :  0.0001154085184680298 결과물 :  [[10.987203]]

# optimizer = Adadelta(lr=0.001)
# loss :  0.0005807600682601333 결과물 :  [[10.953732]]

# optimizer = Adadelta(lr=0.0001)
# loss :  19.440370559692383 결과물 :  [[3.1748626]]

# optimizer = RMSprop(lr=0.001)
# loss :  0.341259241104126 결과물 :  [[10.963417]]

# optimizer = RMSprop(lr=0.0001)
# loss :  0.02834097482264042 결과물 :  [[10.669208]]

# optimizer = SGD(lr=0.001)
# loss :  2.6068848910654197e-06 결과물 :  [[10.998676]]

# optimizer = SGD(lr=0.0001)
# loss :  0.0007156368810683489 결과물 :  [[10.966085]]

# optimizer = Nadam(lr=0.001)
# loss :  0.00035395199665799737 결과물 :  [[10.959557]]

# optimizer = Nadam(lr=0.0001)
# loss :  0.0003148555406369269 결과물 :  [[10.991153]]





model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('loss : ', loss, '결과물 : ', y_pred)

#loss :  0.0012455349788069725 결과물 :  [[10.94237]]
#loss :  0.1434299647808075 결과물 :  [[11.518965]]

#0.1
# loss :  24227.576171875 결과물 :  [[301.3914]]

#0.01
# loss :  0.00041868616244755685 결과물 :  [[11.017192]]

#0.001
# loss :  0.002683038590475917 결과물 :  [[10.912115]]


