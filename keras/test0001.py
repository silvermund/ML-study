import tensorflow as tf
import keras
print(tf.__version__)

print(keras.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())