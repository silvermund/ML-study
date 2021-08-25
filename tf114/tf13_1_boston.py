# 실습
# pip install sklearn

# 실습 : 만들기

# 최종 결론값은 r2_score로 할 것


from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 데이터
tf.set_random_seed(66)

datasets = load_boston()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape) # (506, 13) (506,)


y_data = y_data.reshape(-1,1) # (506, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.2,  random_state = 77)


# 2. 모델링
x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random.normal([13,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')


hypothesis = tf.matmul(x, W) + b

cost = tf.reduce_mean(tf.square(hypothesis-y)) # mse


# 3. 훈련
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)

train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
              feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:
        print(epochs, "cost : ", cost_val, "\n", hy_val)



# 4. 평가, 예측
predicted = sess.run(hypothesis, feed_dict={x:x_test})
r2 = r2_score(y_test, predicted)
print("R2 : ",r2)

sess.close()
