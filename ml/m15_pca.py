import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) #(506, 13) (506,)


# pca = PCA(n_components=12)
# x = pca.fit_transform(x)
# print(x)
# print(x.shape) #(442, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

#2. 모델
model = XGBRegressor()

#3. 훈련
model.fit(x, y)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 :", results)
