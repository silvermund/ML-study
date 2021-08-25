# 실습 : 만들기

# 4. 평가, 예측 <- 코드에 넣고

# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# 최종 결론값은 accuracy_score로 할 것 

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 데이터
tf.set_random_seed(66)

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape) # (569, 30) (569,)

y_data = y_data.reshape(-1,1) # (569, 1)


x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.2,  random_state = 77)


# 2. 모델링
x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random.normal([30,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')


hypothesis = tf.matmul(x, W) + b

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)

train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())



# 3. 훈련
for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
              feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:
        print(epochs, "cost : ", cost_val, "\n", hy_val)



# 4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})

print("==========================================")
print("예측값 : \n", h[0:5], "\n 예측결과값 \n: ", c[0:5], "\n Accuracy \n:", a)

sess.close()

