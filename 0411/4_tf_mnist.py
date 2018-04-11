import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_x = mnist.train.images
train_y = mnist.train.labels

total_epochs = 200
batch_size = 100
learning_rate = 0.0002



def model(x):
    dw1 = tf.get_variable(name = "w1", shape = [784, 256], initializer = tf.random_normal_initializer(0.0, 0.01))
    db1 = tf.get_variable(name = "b1", shape = [256], initializer = tf.random_normal_initializer(0.0, 0.01))
    dw2 = tf.get_variable(name = "w2", shape = [256, 256], initializer = tf.random_normal_initializer(0.0, 0.01))
    db2 = tf.get_variable(name = "b2",  shape = [256], initializer = tf.random_normal_initializer(0.0, 0.01))
    dw3 = tf.get_variable(name = "w3", shape = [256, 10], initializer = tf.random_normal_initializer(0.0, 0.01))
    db3 = tf.get_variable(name = "b3",  shape = [10], initializer = tf.random_normal_initializer(0.0, 0.01))

    hidden1 = tf.nn.relu(tf.matmul(x , dw1) + db1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, dw2) + db2)
    output = tf.nn.softmax(tf.matmul(hidden2, dw3)  + db3)
    return output




g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    y = model(X)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_model = optimizer.minimize(loss)



with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())
    total_batchs = int(train_x.shape[0] / batch_size)

    for epoch in range(total_epochs):
        for batch in range(total_batchs):
            batch_x = train_x[batch * batch_size : (batch+1) * batch_size]
            batch_y = train_y[batch * batch_size : (batch+1) * batch_size]

            sess.run(train_model, feed_dict = {X: batch_x, y_: batch_y})

        if (epoch+1) % 5 == 0 or epoch == 1:
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels}))
