from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1,101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)]) 
x1 = np.transpose(x1)
x2 = np.transpose(x2)
# y1 = np.array([range(1001, 1101)])
# y1 = np.transpose(y1)
y = np.array(range(1001, 1101))

# print(x1.shape, x2.shape, y.shape) # (100, 3) (100, 3) (100,)

#실습
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, random_state=66) #train_size를 넣지 않으면 돌아갈까?

#2-1. 모델1
input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(3, activation='relu', name='dense2')(dense1)
dense3 = Dense(2, activation='relu', name='dense3')(dense2)
output1 = Dense(3, name='output1')(dense3)

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(4, activation='relu', name='dense11')(input2)
dense12 = Dense(4, activation='relu', name='dense12')(dense11)
dense13 = Dense(4, activation='relu', name='dense13')(dense12)
dense14 = Dense(4, activation='relu', name='dense14')(dense13)
output2 = Dense(4, name='output2')(dense14)

merge1 = Concatenate()([output1, output2])
# merge1 = concatenate([output1, output2]) #대문자로 하면 에러!!
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath='./_save/ModelCheckPoint/keras49_mcp.hdf5')

model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8, verbose=1, validation_split=0.2, callbacks=[es, mcp])

model.save('./_save/ModelCheckPoint/keras49_model_save.hdf5')

print("=================================1. 기본 출력 =============================")

# #4. 평가, 예측
results = model.evaluate([x1_test, x2_test], y_test)
y_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
# print(results)
print("loss :", results[0])
print("metrics['mae'] : ", results[1])
print("r2스코어 :", r2)

print("=================================2. load_model ============================")
model2 = load_model('./_save/ModelCheckPoint/keras49_model_save.hdf5')

results = model2.evaluate([x1_test, x2_test], y_test)
# print(results)
y_predict = model2.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)

print("loss :", results[0])
print("metrics['mae'] : ", results[1])
print("r2스코어 :", r2)

print("=================================3. Model Check Point =====================")
model3 = load_model('./_save/ModelCheckPoint/keras49_mcp.hdf5')

results = model3.evaluate([x1_test, x2_test], y_test)
# print(results)
y_predict = model3.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)

print("loss :", results[0])
print("metrics['mae'] : ", results[1])
print("r2스코어 :", r2)

'''
restore_best_weights = false
=================================1. 기본 출력 =============================
1/1 [==============================] - 0s 15ms/step - loss: 1089631.2500 - mae: 1043.4351
loss : 1089631.25
metrics['mae'] :  1043.43505859375
r2스코어 : -1245.1725811286756
=================================2. load_model ============================
1/1 [==============================] - 0s 102ms/step - loss: 1089631.2500 - mae: 1043.4351
loss : 1089631.25
metrics['mae'] :  1043.43505859375
r2스코어 : -1245.1725811286756
=================================3. Model Check Point =====================
1/1 [==============================] - 0s 102ms/step - loss: 1089631.2500 - mae: 1043.4351
loss : 1089631.25
metrics['mae'] :  1043.43505859375
r2스코어 : -1245.1725811286756

restore_best_weights = True
=================================1. 기본 출력 =============================
1/1 [==============================] - 0s 16ms/step - loss: 23.7675 - mae: 4.0497
loss : 23.76752281188965
metrics['mae'] :  4.049692630767822
r2스코어 : 0.9728179249801671
=================================2. load_model ============================
1/1 [==============================] - 0s 100ms/step - loss: 23.7675 - mae: 4.0497
loss : 23.76752281188965
metrics['mae'] :  4.049692630767822
r2스코어 : 0.9728179249801671
=================================3. Model Check Point =====================
1/1 [==============================] - 0s 102ms/step - loss: 23.7675 - mae: 4.0497
loss : 23.76752281188965
metrics['mae'] :  4.049692630767822
r2스코어 : 0.9728179249801671
'''
