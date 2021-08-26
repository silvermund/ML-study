import numpy as np
import tensorflow as tf

# 1. 데이터
tf.set_random_seed(66)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]    # (8, 4)
y_data = [[0, 0, 1],        # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],        # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],        # 0
          [1, 0, 0]]        #(8, 3)

# 만들어

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# sess = tf.compat.v1.Session()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        if step % 200 == 0:
            print(step, cost_val)


# 4. 평가, 예측
    results = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
    print(results, sess.run(tf.math.argmax(results, 1)))


