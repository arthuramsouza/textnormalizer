# Multi-Layer Perceptron Model

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



# plt.imshow(mnist.train.images[1].reshape(28, 28))
# plt.imshow(mnist.train.images[1].reshape(28, 28), cmap='gist_gray')

plt.imshow(mnist.train.images[1].reshape(784, 1), cmap='gist_gray', aspect=0.02)

plt.show()
