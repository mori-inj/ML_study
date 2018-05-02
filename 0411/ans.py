import tensorflow as tf
import numpy as np

n = 1000
d = 100
learning_rate = 0.1

x_data = np.vstack([np.random.normal(0.1, 1, (n//2, d)),np.random.normal(-0.1, 1, (n//2, d))])
y_data = np.hstack([np.ones(n//2), -1.*np.ones(n//2)]).reshape([1000,1])


g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [n, d])
    y = tf.placeholder(tf.float32, [n, 1])
    w0 = tf.Variable(tf.random_normal([d, 1]))
    m = tf.matmul(X, w0)
    y_ = tf.sign(m)
    loss = tf.reduce_mean(tf.maximum(tf.zeros([n,1]), tf.ones([n,1]) - y * m))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(train, feed_dict={X:x_data, y:y_data})
        print(sess.run(loss, feed_dict={X:x_data, y:y_data}))
    y = sess.run(y_, feed_dict={X:x_data, y:y_data})
    
    print(y.shape, y_data.shape)
    pred = tf.equal(y, y_data)
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))
    print(sess.run(pred, feed_dict={X:x_data, y:y_data}))
    print(sess.run(w0))
