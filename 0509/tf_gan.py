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
    l = [random_size, 128, 256, image_size]
    with tf.variable_scope(name_or_scope = "Gen") as scope:
        gw1 = tf.get_variable(name = "w1", shape = [l[0], l[1]], initializer = init)
        gb1 = tf.get_variable(name = "b1", shape = [l[1]], initializer = init)
        gw2 = tf.get_variable(name = "w2", shape = [l[1], l[2]], initializer = init)
        gb2 = tf.get_variable(name = "b2", shape = [l[2]], initializer = init)
        gw3 = tf.get_variable(name = "w3", shape = [l[2], l[3]], initializer = init)
        gb3 = tf.get_variable(name = "b3", shape = [l[3]], initializer = init)

    
    hidden1 = tf.nn.relu(tf.matmul(z , gw1) + gb1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, gw2) + gb2)
    output = tf.nn.sigmoid(tf.matmul(hidden2, gw3) + gb3)
    return output


def discriminator( x , reuse = False):
    l = [image_size, 256, 128, 1]
    with tf.variable_scope(name_or_scope="Dis", reuse=reuse) as scope:
        dw1 = tf.get_variable(name = "w1", shape = [l[0], l[1]], initializer = init)
        db1 = tf.get_variable(name = "b1", shape = [l[1]], initializer = init)
        dw2 = tf.get_variable(name = "w2", shape = [l[1], l[2]], initializer = init)
        db2 = tf.get_variable(name = "b2",  shape = [l[2]], initializer = init)
        dw3 = tf.get_variable(name = "w3", shape = [l[2], l[3]], initializer = init)
        db3 = tf.get_variable(name = "b3",  shape = [l[3]], initializer = init)

    
    hidden1 = tf.nn.relu(tf.matmul(x , dw1) + db1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, dw2) + db2)
    output = tf.nn.sigmoid(tf.matmul(hidden2, dw3)  + db3)
    return output

def random_noise(batch_size):
return np.random.normal(size=[batch_size , random_size])
