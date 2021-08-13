import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)


pca = PCA(n_components=10)
x = pca.fit_transform(x)
# print(x)
print(x.shape) #(442, 7)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.94)+1)

plt.plot(cumsum)
plt.grid()
plt.show()


'''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

#2. 모델
model = XGBRegressor()

#3. 훈련
model.fit(x, y)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 :", results)
'''



