# 실습
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]


# y = wx +  b

import tensorflow as tf
tf.compat.v1.set_random_seed(77)

# x_train = [1,2,3] # w=1, b=0
# y_train = [3,5,7]

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# W = tf.Variable(1, dtype = tf.float32) # 랜던하게 내맘대로 넣어준
# b = tf.Variable(1, dtype = tf.float32) # 초기값

#random_normal -> 정규분포화
W = tf.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32) # 랜던하게 내맘대로 넣어준
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32) # 초기값

hypothesis = x_train * W + b    # 모델구현
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b], 
                        feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
    if step % 20 ==0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(step, loss_val, W_val, b_val)


# predict 하는 코드를 추가할 것
# x_test라는 placeholder 생성!!!

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

hyperthesis_p = x_test * W_val + b_val

pred1 = sess.run(hyperthesis_p, feed_dict={x_test:[4]})
pred2 = sess.run(hyperthesis_p, feed_dict={x_test:[5,6]})
pred3 = sess.run(hyperthesis_p, feed_dict={x_test:[6,7,8]})

print("predict [4] :",pred1)
print("predict [5, 6] :",pred2)
print("predict [6, 7, 8] :",pred3)
