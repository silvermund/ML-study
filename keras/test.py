from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터

x = np.array(range(100))
y = np.array(range(1, 101))

# x_train = x[:70]
# y_train = y[:70]
# x_test = x[-30:]
# y_test = y[70:]

# print(x_train.shape, y_train.shape) # (70,) (70,)
# print(y_train.shape, y_test.shape)  # (30,) (30,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

print(x_test)
print(y_test)
print(x_train)
print(y_train)
