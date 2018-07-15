# model
K = 4
L = 8
M = 12

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.5), name='w1')
b1 = tf.Variable(tf.truncated_normal([K], stddev=0.5), name='b1')

W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.5), name='w2')
b2 = tf.Variable(tf.truncated_normal([L], stddev=0.5), name='b2')

W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.5), name='w3')
b3 = tf.Variable(tf.truncated_normal([M], stddev=0.5), name='b3')

N = 200

W4 = tf.Variable(tf.truncated_normal([7*7*M, N], stddev=0.5), name='w4')
b4 = tf.Variable(tf.truncated_normal([N], stddev=0.5), name='b4')

W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.5), name='w5')
b5 = tf.Variable(tf.truncated_normal([10], stddev=0.5), name='b5')

# summaries of weights
if True:
    tf.summary.histogram("weights/w1", W1)
    tf.summary.histogram("biases/b1", b1)
    tf.summary.histogram("weights/w2", W2)
    tf.summary.histogram("biases/b2", b2)
    tf.summary.histogram("weights/w3", W3)
    tf.summary.histogram("biases/b3", b3)
    tf.summary.histogram("weights/w4", W4)
    tf.summary.histogram("biases/b4", b4)
    tf.summary.histogram("weights/w5", W5)
    tf.summary.histogram("biases/b5", b5)

# convolution-layers
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') + b1, name='conv_layer_1')
Y2 = tf.nn.sigmoid(tf.nn.conv2d(Y1, W2, strides=[1,2,2,1], padding='SAME') + b2, name='conv_layer_2')
Y3 = tf.nn.sigmoid(tf.nn.conv2d(Y2, W3, strides=[1,2,2,1], padding='SAME') + b3, name='conv_layer_3')

# fully-connected-layer
YY = tf.reshape(Y3, shape=[-1, 7*7*M], name='fcc')
YY = tf.nn.dropout(YY, keep_prob)

# hidden-layer
Y4 = tf.nn.sigmoid(tf.matmul(YY, W4) + b4, name='hidden_layer')
Y4 = tf.nn.dropout(Y4, keep_prob)

# model-output
Y = tf.nn.softmax(tf.matmul(Y4, W5) + b5, name='ouput_layer')

# add weights to collection
tf.add_to_collection('conv_weights', W1)
tf.add_to_collection('conv_weights', W2)
tf.add_to_collection('conv_weights', W3)

# add output to collection
tf.add_to_collection('conv_output', Y1)
tf.add_to_collection('conv_output', Y2)
tf.add_to_collection('conv_output', Y3)

# cost
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y), name='cross_entropy')
tf.summary.scalar("cross_entropy", cross_entropy)

# accuracy
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='accuracy')
tf.summary.scalar("accuracy", accuracy)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

# summaries of h_params
tf.summary.scalar("number_of_epochs", n_epochs)
tf.summary.scalar("mini_batch_size", batch_size)
tf.summary.scalar("learning_rate", optimizer._lr)
