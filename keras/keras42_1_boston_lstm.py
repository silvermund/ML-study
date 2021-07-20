from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv1D, Flatten, MaxPool1D, Dropout, GlobalAveragePooling1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
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
# model.add(Conv1D(filters=16, kernel_size=2, padding='same', input_shape=(13,1)))
# model.add(Dropout(0.2))
# model.add(Conv1D(16, 2, padding='same', activation='relu'))   
# model.add(MaxPool1D())

# model.add(Conv1D(64, 2, padding='same', activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Conv1D(64, 2, padding='same', activation='relu')) 
# model.add(MaxPool1D())

# model.add(Conv1D(256, 2, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv1D(256, 2, padding='same', activation='relu'))  
# model.add(MaxPool1D())

# model.add(GlobalAveragePooling1D())
# model.add(Dense(1))

model.add(LSTM(16, activation='relu', input_length=13, input_dim=1))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)


start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.25, callbacks=[es])
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

lstm
time =  57.62958550453186
loss :  20.580474853515625
R^2 score :  0.7371380732693589
'''

