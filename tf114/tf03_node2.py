# 실습
# 덧셈 node3
# 뺼셈 node4
# 곱셈 node5
# 나눗셈 node6
# 만들어

import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.divide(node1, node2)

sess = tf.Session()
print('add', sess.run(node3))
print('subtract',sess.run(node4))
print('multiply', sess.run(node5))
print('divide', sess.run(node6))

