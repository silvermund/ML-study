import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

#print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) #(70000, 28, 28)
x = x.reshape(70000, 28*28)

# 실습
# pca를 통해 0.95 이상인게 몇개?

pca = PCA(n_components=784)
x = pca.fit_transform(x)

print(x.shape)

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
# print(cumsum)

print(np.argmax(cumsum >= 1)+1)

plt.plot(cumsum)
plt.grid()
plt.show()