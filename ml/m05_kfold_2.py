from numpy.core.numeric import cross
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
# from sklearn import datasets
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

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

# model = LinearSVC()
# Acc :  [1.         0.95833333 0.95833333 1.         0.91666667]
# 평균 Acc :  0.9667

# model = SVC()
# Acc :  [0.95833333 1.         0.95833333 1.         0.875     ]
# 평균 Acc :  0.9583

# model = KNeighborsClassifier()
# Acc :  [0.91666667 1.         0.95833333 1.         0.95833333]
# 평균 Acc :  0.9667

# model = LogisticRegression()
# Acc :  [0.95833333 1.         0.95833333 1.         0.91666667]
# 평균 Acc :  0.9667

# model = RandomForestClassifier()
# Acc :  [0.95833333 0.95833333 0.95833333 1.         0.875     ]
# 평균 Acc :  0.95

# model = DecisionTreeClassifier()
# Acc :  [0.95833333 0.95833333 0.91666667 1.         0.875     ]
# 평균 Acc :  0.9417


#3. 컴파일, 훈련

#4. 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("Acc : ", scores)
print("평균 Acc : ", round(np.mean(scores),4))

# results = model.score(x_test, y_test)
# print(results)
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)



# print("=================예측===============")
# print(y_test[:5])
# y_predict2 = model.predict(x_test[:5])
# print(y_predict2)


