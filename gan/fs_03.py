import torch
import torch.nn as nn
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('../_data/fs/mnist_data/mnist_train.csv', header=None)


# print(df.head())


# df.info()


# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 60000 entries, 0 to 59999
# Columns: 785 entries, 0 to 784
# dtypes: int64(785)
# memory usage: 359.3 MB


# get data from dataframe
row = 13
data = df.iloc[row]

# label is the first value
label = data[0]

# image data is the remaining 784 values
img = data[1:].values.reshape(28,28)
plt.title("label = " + str(label))
plt.imshow(img, interpolation='none', cmap='Blues')
plt.show()

