from numpy.core.numeric import cross
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
# from sklearn import datasets
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

datasets = load_iris()

# print(datasets.DESCR)
# print(datasets.feature_names)

#1. 데이터
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(150, 4) (150,)

# print(y) # y가 0,1,2
# print(np.unique(y))


# y = to_categorical(y)
# print(y.shape) # (150, 3)

# 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)



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

# model = LinearSVC()
# Acc :  [0.96666667 0.96666667 1.         0.9        1.        ]
# 평균 Acc :  0.9667

# model = SVC()
# Acc :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]
# 평균 Acc :  0.9667

# model = KNeighborsClassifier()
# Acc :  [0.96666667 0.96666667 1.         0.9        0.96666667]
# 평균 Acc :  0.96

# model = KNeighborsRegressor()
# Acc :  [0.93758389 0.972      0.9942029  0.85572519 0.97487923]
# 평균 Acc :  0.9469

# model = LogisticRegression()
# Acc :  [1.         0.96666667 1.         0.9        0.96666667]
# 평균 Acc :  0.9667

# model = RandomForestClassifier()
# Acc :  [0.96666667 0.96666667 1.         0.9        0.96666667]
# 평균 Acc :  0.96

# model = DecisionTreeClassifier()
# Acc :  [0.96666667 0.96666667 1.         0.9        0.93333333]
# 평균 Acc :  0.9533

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]},
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]

# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1)

model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1)


# model = SVC()


#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
print("best_score_ : ", model.best_score_)

print("model.score: ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("정답률 : ", accuracy_score(y_test, y_predict))

print("걸린시간 : ", time.time() - start_time)

# scores = cross_val_score(model, x, y, cv=kfold)
# print("Acc : ", scores)
# print("평균 Acc : ", round(np.mean(scores),4))

# results = model.score(x_test, y_test)
# print(results)
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)



# print("=================예측===============")
# print(y_test[:5])
# y_predict2 = model.predict(x_test[:5])
# print(y_predict2)

# GridSearchCV
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# best_score_ :  0.9714285714285715
# model.score:  0.9555555555555556
# 정답률 :  0.9555555555555556
# 걸린시간 :  0.10899710655212402

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# best_score_ :  0.9714285714285715
# model.score:  0.9555555555555556
# 정답률 :  0.9555555555555556
# 걸린시간 :  0.06600093841552734