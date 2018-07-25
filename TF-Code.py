import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utility

# import DataSet
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False, validation_size=0)

# Input 28-by-28 pixels images of GRAYSCALE
X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')

# Output in 'one-hot-encoding', 10 classes
Y_ = tf.placeholder(tf.float32, [None, 10], name='Y_')

# placeholders for hyper parameters
keep_prob = tf.placeholder(tf.float32, [], name='dropout_probability')
learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
n_epochs = 2000
batch_size = 100
