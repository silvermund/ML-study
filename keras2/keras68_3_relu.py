import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# relu의 y값은 음수가 없다.

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 과제
# elu, selu, reaky, relu...
# 68_3_2, 3, 4로 만들것!!!

