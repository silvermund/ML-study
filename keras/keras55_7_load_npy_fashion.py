import numpy as np

x_train_data = np.load('./_save/_npy/k55_x_train_fashion_mnist.npy')
x_test_data = np.load('./_save/_npy/k55_x_train_fashion_mnist.npy')
y_train_data = np.load('./_save/_npy/k55_y_train_fashion_mnist.npy')
y_test_data = np.load('./_save/_npy/k55_x_train_fashion_mnist.npy')


print(type(x_train_data), type(x_test_data)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(y_train_data), type(y_test_data)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>


# print(x_data.shape, y_data.shape) #(150, 4) (150,)