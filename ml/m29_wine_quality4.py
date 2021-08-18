import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from xgboost import XGBClassifier

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)

print(datasets.head())
print(datasets.shape) # (4898, 12)
print(datasets.describe())

datasets = datasets.values
print(type(datasets))  # <class 'numpy.ndarray'>
print(datasets.shape) # (4898, 12)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x = datasets[:, :11]
y = datasets[:, 11]
print(y.shape) #(4898,)

newlist = []
for i in list(y):
    if i<= 4:
        newlist +=[0]
    elif i<=7:
        newlist +=[1]
    else:
        newlist +=[2]

y = np.array(newlist)
print(y.shape) #(4898,)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)

print("accuracy : ", score) 
# accuracy :  0.6816326530612244 -> 0.9469387755102041
