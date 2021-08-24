import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly()) # True


tf.compat.v1.disable_eager_execution() #즉시실행모드

print(tf.executing_eagerly()) # False

'''
# print('hello world')

hello = tf.constant("Hello World")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))

# b'Hello World'

'''