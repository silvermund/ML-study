# 인공지능의 겨울을 극복하자
# perceptron -> mlp

import tensorflow as tf
tf.compat.v1.set_random_seed(66)

# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] # (4, 2)
y_data = [[0],[1],[1],[0]]            # (4, 1)

# 2. 모델링
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 히든레이어 1
W1 = tf.Variable(tf.random.normal([2,3]), name='weight')
b1 = tf.Variable(tf.random.normal([3]), name='bias')

# hypothesis = x * w + b
layer1 = tf.sigmoid(tf.matmul(x, W1) + b1)

# # 히든레이어 2
W2 = tf.Variable(tf.random.normal([3,3]), name='weight')
b2 = tf.Variable(tf.random.normal([3]), name='bias')

layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# 아웃풋레이어
W3 = tf.Variable(tf.random.normal([3,1]), name='weight')
b3 = tf.Variable(tf.random.normal([1]), name='bias')


# hypothesis = x * w + b
hypothesis = tf.sigmoid(tf.matmul(layer2, W2) + b3)

# cost = tf.reduce_mean(tf.square(hypothesis-y)) # mse
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)

train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 3. 훈련
for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
              feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:
        print(epochs, "cost : ", cost_val, "\n", hy_val)


# 4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print("==========================================")
print("예측값 : \n", hy_val, "\n 예측결과값 \n: ", c, "\n Accuracy \n:", a)

sess.close()
