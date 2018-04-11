import tensorflow as tf

learning_rate = 0.001

g = tf.Graph()

with g.as_default():
    X = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
    Y = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
    Z = (X-1)*(X-1) + (Y+1)*(Y+1)
    loss = tf.reduce_mean(Z)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run(train)

    x,y = sess.run([X,Y])

    print(x,y)

