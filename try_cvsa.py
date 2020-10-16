# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 07:40:56 2019

@author: cvegas
"""

import gzip
import pickle


with gzip.open('mnist.pkl.gz', 'rb') as f:
#    train_set, valid_set, test_set = pickle.load(f)
    train_set, valid_set, test_set = pickle.load(f, encoding='iso-8859-1')
    
train_x, train_y = train_set

import matplotlib.cm as cm
import matplotlib.pyplot as plt

a=0 
while a <500:
    plt.imshow(train_x[a].reshape((28, 28)), cmap=cm.Greys_r)
    plt.show()
    a=a+1