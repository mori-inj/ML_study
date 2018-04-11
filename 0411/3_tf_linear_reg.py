import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(201):
	sess.run(train)
	if step % 20 == 0:
	    print(step, sess.run(W), sess.run(b))


