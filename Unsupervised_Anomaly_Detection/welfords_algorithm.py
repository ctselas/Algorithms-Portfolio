# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 09:36:14 2019

@author: Christos.TSELAS , Data Analyst - Machine Learning Engineer
ctselas@gmail.com

Welford's Online algorithm for 2 dimensional data (#samples, #features)

* computes mean and (estimated) variance and updates them as new data are created
* normalize the data in relation to the previous
"""
import numpy as np

class Welford(object):
    def __init__(self, feature_size=1):
        self.n = 0
        self.mean = np.zeros((1, feature_size))
        self.mean_diff = np.zeros((1, feature_size))
        self.var = np.zeros((1, feature_size))
        
    def observe(self, data):
        'Update mean and variance'
        sample_size = data.shape[0]
        last_mean = list(self.mean[0])
        
        for i, value in enumerate(data):
            self.mean += (value-self.mean)/(self.n+i+1)
        self.n += sample_size
        
        a = np.matrix((data-np.repeat(np.matrix(last_mean), sample_size, axis=0)))
        b = np.matrix((data-np.repeat(self.mean, sample_size, axis=0)))
        
        self.mean_diff += np.diag(a.T*b)
        self.var = np.clip(self.mean_diff/self.n, 1e-2, self.mean_diff/self.n)
        
    def normalize(self, data):
        'Normalize data'
        obs_std = np.sqrt(self.var)
        return (data - self.mean)/obs_std
    
    
# Example
normalizer = Welford((data.shape[1])) # initialize

normalizer.observe(data) # pass data to compute new mean, variance
normalized_data = normalizer.normalize(data) # normalize data
