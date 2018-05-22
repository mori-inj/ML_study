import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

train_x = []
base_path = './Pokemon/'
for filename in os.listdir(base_path):
    filename, fileext = os.path.splitext(filename)
    if fileext == '.jpg':
        img = plt.imread(base_path+filename+'.jpg')
        train_x.append(img)
        
train_x = np.asarray(train_x)
print(train_x.shape)
train_x = train_x.reshape([train_x.shape[0], -1])
print(train_x.shape)

