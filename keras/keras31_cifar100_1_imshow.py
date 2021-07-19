from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import OneHotEncoder

#이미지가  32, 32, 3 칼라

# 완성하시오

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

