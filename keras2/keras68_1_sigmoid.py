import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
print(len(x))

y = sigmoid(x)
# y는 0~1


plt.plot(x, y)
plt.grid()
plt.show()