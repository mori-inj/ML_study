import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

train_x = mnist.train.images
train_y = mnist.train.labels

print(train_x.shape, train_y.shape)

total_epochs = 450
batch_size = 100
learning_rate = 0.0001
random_size = 100
image_size = 28*28

init = tf.random_normal_initializer(mean=0.0, stddev=0.01)

def generator( z , reuse = False ):
    with tf.variable_scope(name_or_scope = "Gen") as scope:
        pass


def discriminator( x , reuse = False):
    with tf.variable_scope(name_or_scope="Dis", reuse=reuse) as scope:
        pass

def random_noise(batch_size):
    return np.random.normal(size=[batch_size , random_size])




g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, 784])
    Z = tf.placeholder(tf.float32, [None, random_size])

    ??? = generator(???)
    ??? = discriminator(???, False)
    ??? = discriminator(???, True)

    g_loss = tf.reduce_mean( tf.log(???) )
    d_loss = tf.reduce_mean( tf.log(???) + tf.log(???) )

    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if "Gen" in var.name]
    d_vars = [var for var in t_vars if "Dis" in var.name]
    optimizer = tf.train.AdamOptimizer(learning_rate)
    g_train = optimizer.minimize(-g_loss, var_list= g_vars)
    d_train = optimizer.minimize(-d_loss, var_list = d_vars)



with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())
    total_batchs = int(train_x.shape[0] / batch_size)

    for epoch in range(total_epochs):
        for batch in range(total_batchs):
            batch_x = train_x[batch * batch_size : (batch+1) * batch_size]
            batch_y = train_y[batch * batch_size : (batch+1) * batch_size]
            noise = random_noise(batch_size)

            sess.run(g_train , feed_dict = {Z : noise})
            sess.run(d_train, feed_dict = {X : batch_x , Z : noise})

            gl, dl = sess.run([g_loss, d_loss], feed_dict = {X : batch_x , Z : noise})


        if epoch % 2 == 0:
            print("======= Epoch : ", epoch , " =======")
            print("generator: " , -gl )
            print("discriminator: " , -dl )


        samples = 20
        if epoch % 2 == 0:
            sample_noise = random_noise(samples)
            gen = sess.run(fake_x , feed_dict = { Z : sample_noise})

            for i in range(samples):
                img = gen[i].reshape([28,28])
                mpimg.imsave('./epoch/epoch'+str(epoch)+'_'+str(i)+'.png', img, format='png')
