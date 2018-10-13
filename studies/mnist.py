# Multi-Layer Perceptron Model

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Suppress AVX2 warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = input_data.read_data_sets("MNIST_data/", one_hot=True)

plt.imshow(data.train.images[1].reshape(784, 1), cmap='gist_gray', aspect=0.02)
plt.show()

# model
x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
y_true = tf.placeholder(tf.float32, shape=[None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        batch_x, batch_y = data.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))

    # accuracy
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

    print(sess.run(acc, feed_dict={x: data.test.images, y_true: data.test.labels}))
