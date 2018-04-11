import tensorflow as tf
import numpy as np

n = 1000
d = 100
learning_rate = 0.01

x_data = np.vstack([np.random.normal(0.1, 1, (n//2, d)),np.random.normal(-0.1, 1, (n//2, d))])
y_data = np.hstack([np.ones(n//2), -1.*np.ones(n//2)])


g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [n, d])
    y = tf.placeholder(tf.float32, [1,n])
    w0 = tf.Variable(tf.random_normal([d]))
    m = w0 * X
    loss = tf.reduce_mean(tf.maximum(tf.zeros([n,d]), tf.ones([n,d]) - tf.matmul(y,m)))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(300):
        sess.run(train, feed_dict={X:x_data, y:[y_data]})
        print(sess.run(loss, feed_dict={X:x_data, y:[y_data]}))

    print(sess.run(w0))
