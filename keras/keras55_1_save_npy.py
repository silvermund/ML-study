from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

datasets = load_iris()

x_data_iris = datasets.data
y_data_iris = datasets.target

print(type(x_data_iris), type(y_data_iris)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data_iris)
np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data_iris)


# 실습 8분 동안 한다
# 보스톤, 캔서, 디아벳까지 npy로 세이브

datasets = load_boston()

x_data_boston = datasets.data
y_data_boston = datasets.target

print(type(x_data_boston), type(y_data_boston)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_boston.npy', arr=x_data_boston)
np.save('./_save/_npy/k55_y_data_boston.npy', arr=y_data_boston)


datasets = load_breast_cancer()

x_data_cancer = datasets.data
y_data_cancer = datasets.target

print(type(x_data_cancer), type(y_data_cancer)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_cancer.npy', arr=x_data_cancer)
np.save('./_save/_npy/k55_y_data_cancer.npy', arr=y_data_cancer)


datasets = load_diabetes()

x_data_diabetes = datasets.data
y_data_diabetes = datasets.target

print(type(x_data_diabetes), type(y_data_diabetes)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_diabetes.npy', arr=x_data_diabetes)
np.save('./_save/_npy/k55_y_data_diabetes.npy', arr=y_data_diabetes)


datasets = load_wine()

x_data_wine = datasets.data
y_data_wine = datasets.target

print(type(x_data_wine), type(y_data_wine)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data_wine)
np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data_wine)


####################################################################################################

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()


# print(type(x_train, y_train), type(x_test, y_test)) #<class 'tuple'> <class 'tuple'>


np.save('./_save/_npy/k55_x_train_mnist.npy', arr=x_train_mnist)
np.save('./_save/_npy/k55_x_test_mnist.npy', arr=x_test_mnist)
np.save('./_save/_npy/k55_y_train_mnist.npy', arr=y_train_mnist)
np.save('./_save/_npy/k55_y_test_mnist.npy', arr=y_test_mnist)


(x_train_fashion_mnist, y_train_fashion_mnist), (x_test_fashion_mnist, y_test_fashion_mnist) =  fashion_mnist.load_data()



np.save('./_save/_npy/k55_x_train_fashion_mnist.npy', arr=x_train_fashion_mnist)
np.save('./_save/_npy/k55_x_test_fashion_mnist.npy', arr=x_test_fashion_mnist)
np.save('./_save/_npy/k55_y_train_fashion_mnist.npy', arr=y_train_fashion_mnist)
np.save('./_save/_npy/k55_y_test_fashion_mnist.npy', arr=y_test_fashion_mnist)


(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) =  cifar10.load_data()


np.save('./_save/_npy/k55_x_train_cifar10.npy', arr=x_train_cifar10)
np.save('./_save/_npy/k55_x_test_cifar10.npy', arr=x_test_cifar10)
np.save('./_save/_npy/k55_y_train_cifar10.npy', arr=y_train_cifar10)
np.save('./_save/_npy/k55_y_test_cifar10.npy', arr=y_test_cifar10)


(x_train_cifar100, y_train_cifar100), (x_test_cifar100, y_test_cifar100) =  cifar100.load_data()

np.save('./_save/_npy/k55_x_train_cifar100.npy', arr=x_train_cifar100)
np.save('./_save/_npy/k55_x_test_cifar100.npy', arr=x_test_cifar100)
np.save('./_save/_npy/k55_y_train_cifar100.npy', arr=y_train_cifar100)
np.save('./_save/_npy/k55_y_test_cifar100.npy', arr=y_test_cifar100)

