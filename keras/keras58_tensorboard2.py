# C:\Users\clife>d:

# D:\>f:

# F:\>cd study

# F:\study>cd _save

# F:\study\_save>cd _graph

# F:\study\_save\_graph>tensorboard --logdir=.



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,6,7,8,9,10])
x_pred = [6]

#2. 모델 구성
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x, y, epochs=100, batch_size=1, callbacks=[es, tb], validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('6의 예측값 : ', result)



