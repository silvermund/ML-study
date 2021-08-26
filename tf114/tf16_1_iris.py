# 실습
# 만들기
# accuracy 넣을 것

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# 1. 데이터
tf.set_random_seed(66)

datasets = load_iris()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape) # (150, 4) (150,)

y_data = y_data.reshape(-1,1) # (150, 1)

encoder = OneHotEncoder()
encoder.fit(y_data)
y_data = encoder.transform(y_data).toarray()

print(y_data.shape) #(150, 3)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.2, random_state = 77)


# 2. 모델링
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random.normal([4, 3]), name='weight')
b = tf.Variable(tf.random.normal([1, 3]), name='bias')

# hypothesis = tf.matmul(x, W) + b
hypothesis = tf.nn.softmax(tf.matmul(x, W) + b) 

# categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# 3. 훈련
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(5001):
    _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
    if epochs % 200 == 0:
            print(epochs, cost_val)


# 4. 평가, 예측
predicted = sess.run(hypothesis, feed_dict={x:x_test})
print(predicted, sess.run(tf.argmax(predicted, 1)))

y_pred = sess.run(hypothesis, feed_dict={x:x_test})
y_pred = np.argmax(y_pred, axis= 1)
y_test = np.argmax(y_test, axis= 1)

print('acc_score : ', accuracy_score(y_test, y_pred))


sess.close()
