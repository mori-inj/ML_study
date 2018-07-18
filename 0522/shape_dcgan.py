import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

train_x = []
base_path = './Shape/'
for filename in os.listdir(base_path):
    filename, fileext = os.path.splitext(filename)
    if fileext == '.png':
        img = plt.imread(base_path+filename+'.png')
        train_x.append(np.zeros([128,128,4]))
        train_x[-1][0:100,0:100,:] = img
        
train_x = np.asarray(train_x)
print(train_x.shape)

total_epochs = 450
batch_size = 100
learning_rate = 0.001
random_size = 128

init = tf.random_normal_initializer(mean=0.0, stddev=0.01)



def generator( z , reuse = False ):
    l = [64, 32, 4]

    with tf.variable_scope(name_or_scope = 'Gen') as scope:
        output = tf.layers.dense(z, random_size * 4 * 4, name='fc')
        output = tf.reshape(output, [-1, 4, 4, random_size])

        output = tf.layers.conv2d_transpose(output, l[0], 5, strides=4, padding='SAME', name='dcv0') # 16*16
        output = tf.nn.relu(tf.layers.batch_normalization(output, name='bn0'))
        output = tf.layers.conv2d_transpose(output, l[1], 5, strides=4, padding='SAME', name='dcv2') # 64*64
        output = tf.nn.relu(tf.layers.batch_normalization(output, name='bn2'))
        output = tf.layers.conv2d_transpose(output, l[2], 5, strides=2, padding='SAME', name='dcv4') # 256*256
        output = tf.nn.sigmoid(output)

    return output


def discriminator( x , reuse = False):
    l = [32, 64, 128]

    with tf.variable_scope(name_or_scope = 'Dis', reuse=reuse) as scope:
        output = tf.layers.conv2d(x, l[0], 5, strides=2, padding='SAME', name='cv0') # 128*128 8
        output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, name='bn0'))
        output = tf.layers.conv2d(output, l[1], 5, strides=4, padding='SAME', name='cv2') # 32*32 32
        output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, name='bn2'))
        output = tf.layers.conv2d(output, l[2], 5, strides=4, padding='SAME', name='cv4') # 8*8 128
        output = tf.nn.leaky_relu(tf.layers.batch_normalization(output), name='bn4')
        
        output = tf.contrib.layers.flatten(output)
        output = tf.layers.dense(output, 1, name='fc')
        output = tf.nn.sigmoid(output)


    return output

def random_noise(batch_size):
    return np.random.normal(size=[batch_size , random_size])




g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, 128, 128, 4])
    Z = tf.placeholder(tf.float32, [None, random_size])

    fake_x = generator(Z)
    result_of_fake = discriminator(fake_x)
    result_of_real = discriminator(X , True)

    g_loss = tf.reduce_mean( tf.log(result_of_fake) )
    d_loss = tf.reduce_mean( tf.log(result_of_real) + tf.log(1 - result_of_fake) )

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
                img = gen[i].reshape([128,128,4])
                mpimg.imsave('./epoch/epoch'+str(epoch)+'_'+str(i)+'.png', img, format='png')
