from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
# from sklearn import datasets
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
import matplotlib.pyplot as plt
import time

datasets = load_iris()


#1. 데이터
x = datasets.data
y = datasets.target


# 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

# scaler = StandardScaler()
# scaler.fit(x_train) # 훈련
# x_train = scaler.transform(x_train) # 변환
# x_test = scaler.transform(x_test)

#2. 모델 구성
# model = LinearSVC()
# accuracy :  0.9333333373069763

# model = SVC()
# accuracy_score :  0.9555555555555556

# model = KNeighborsClassifier()
# accuracy_score :  0.9777777777777777

# model = KNeighborsRegressor()

# model = LogisticRegression()
# accuracy_score :  0.9777777777777777


# model = RandomForestClassifier()
# accuracy_score :  0.9111111111111111

# model = DecisionTreeClassifier()
# accuracy_score :  0.9111111111111111


# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(4,))) 
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(3, activation='softmax'))


model = make_pipeline(MinMaxScaler(), RandomForestClassifier()) 


#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

# hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es])


# print(hist.history.keys())
# print("================================")
# print(hist.history['loss'])
# print("================================")
# print(hist.history['val_loss'])
# print("================================")



#4. 평가, 예측
# results = model.score(x_test, y_test)
# print(results)
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)



# print("=================예측===============")
# print(y_test[:5])
# y_predict2 = model.predict(x_test[:5])
# print(y_predict2)

# print("최적의 매개변수 : ", model.best_estimator_)
# print("best_params_ : ", model.best_params_)
# print("best_score_ : ", model.best_score_)

print("model.score: ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("r2_score : ", r2_score(y_test, y_predict))

print("걸린시간 : ", time.time() - start_time)


'''
print("==============평가, 예측=============")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

print("=================예측===============")
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)

# plt.plot(hist.history['loss'])  # x:epoch, / y:hist.history['loss']
# plt.plot(hist.history['val_loss'])

# plt.title("loss, val_loss")
# plt.xlabel('epoch')
# plt.ylabel('loss, val_loss')
# plt.legend(['train loss','val loss'])
# plt.show()

'''
# loss :  1.138843059539795
# accuracy :  0.9333333373069763

# model.score:  0.9111111111111111
# r2_score :  0.8566878980891719
# 걸린시간 :  0.09702777862548828