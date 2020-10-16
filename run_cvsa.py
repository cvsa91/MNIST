# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:13:09 2019

@author: cvegas
"""
import numpy as np

layers=3

# =============================================================================
# 
# to train and test the data
# 
# 
# =============================================================================

#import mnist_loader
#training_data, validation_data, test_data = \
#mnist_loader.load_data_wrapper()
#
#import network
##[neurons layer 1,nl2,nl3]
#net = network.Network([784, 30, 10])
#
##[epochs,mini_mbatch,learning rate]
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#
#
#
#b=  net.biases
#w=  net.weights
#
#np.savez('wyb_cvsa', b=b, w=w)

# =============================================================================
# 
# to load saved weights and biases
# 
# =============================================================================
#

t=np.load('wyb.npz',allow_pickle=True)

br=t['b']
b0=br[0]
b1=br[1]

wr=t['w']
w0=wr[0]
w1=wr[1]


# =============================================================================
# 
# to get 1 random input image
# 
# 
# =============================================================================
import pickle
import gzip
import random
import numpy

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data)

def load_data_wrapper():
    rdn_image=random.randint(0,9999)
    tr_d, va_d, te_d = load_data()

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_input = test_inputs[rdn_image]
    te_d1=te_d[1]
    test_data = test_input, te_d1[rdn_image]
    return (test_data)


import matplotlib.cm as cm
import matplotlib.pyplot as plt

data=load_data_wrapper()

plt.imshow(data[0].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()
print ("The objective number is ", data[1])


# =============================================================================
# 
# to EVALUATE  1 input (first of all)
# 
# 
# =============================================================================


for l in range(0,layers-1):    
    if (l == 0):
            activation = np.dot(wr[l], data[0])+br[l]
    else:
            activation = np.dot(wr[l], activation)+br[l]
            
result1=np.argmax(activation)
print ("The guess is: " , result1)



