import numpy as np
import os
import tensorflow as tf

# Suppress AVX2 warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Operations with Constants

x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition: ', sess.run(x + y))
    print('Subtraction: ', sess.run(x - y))
    print('Multiplication: ', sess.run(x * y))
    print('Division: ', sess.run(x / y))

# Operations with Placeholders

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

add = tf.add(x, y)
sub = tf.subtract(x, y)
mul = tf.multiply(x, y)

d = {x: 20, y: 30}

with tf.Session() as sess:
    print('Operations with Placeholders')
    print('addition', sess.run(add, feed_dict=d))
    print('subtraction', sess.run(sub, feed_dict=d))
    print('addition', sess.run(mul, feed_dict=d))

# Operations with Matrices

a = np.array([[5.0, 5.0]])
b = np.array([[2.0], [2.0]])

print(a)
print(a.shape)
print(b)
print(b.shape)

mat1 = tf.constant(a)
mat2 = tf.constant(b)

matrix_multi = tf.matmul(mat1, mat2)

with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)
