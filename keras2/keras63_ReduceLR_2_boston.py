from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv1D, Flatten, MaxPool1D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
import time



datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=5)

print(x.shape) #(506, 13)
print(y.shape) #(506,)
# print(datasets.feature_names) ## ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] 13의 요소
# print(datasets.DESCR)

print(x_train.shape) #(404, 13)
print(x_test.shape) #(102, 13)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 변환
x_test = scaler.transform(x_test)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape) #(404, 13, 1)
print(x_test.shape) #(102, 13, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, padding='same', input_shape=(13,1)))
model.add(Dropout(0.2))
model.add(Conv1D(16, 2, padding='same', activation='relu'))   
model.add(MaxPool1D())

model.add(Conv1D(64, 2, padding='same', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', activation='relu')) 
model.add(MaxPool1D())

model.add(Conv1D(256, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(256, 2, padding='same', activation='relu'))  
model.add(MaxPool1D())

# model.add(Flatten())    
# model.add(Dense(128, activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))  
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu')) 
model.add(GlobalAveragePooling1D())
model.add(Dense(1))


#3. 컴파일, 훈련

optimizer = Adam(lr=0.001)

model.compile(loss='mse', optimizer=optimizer, metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, validation_split=0.25, callbacks=[es, reduce_lr])
end_time = time.time() - start_time


#4. 평가, 예측
y_predict = model.predict([x_test])

loss = model.evaluate(x_test, y_test)
print("time = ", end_time)
print('loss : ', loss)
r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)


'''
time =  6.529106378555298
loss :  16.59917449951172
R^2 score :  0.7879888263311785

time =  11.219877004623413
loss :  12.410390853881836
R^2 score :  0.8414896283873469

reduce_lr
time =  14.700733661651611
loss :  [86.20774841308594, 0.0]
R^2 score :  -0.1010791460067959
'''

