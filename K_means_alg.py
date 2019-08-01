# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:02:48 2019

@author: gakel
"""

#generate simulated data


import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


mu1 = [3, 4]
sigma1 = [[5, .3], [.3, 1]]

mu2 = [7, 7]
sigma2 = [[5, .5], [.5, 4]]

sample1 = np.random.multivariate_normal(mu1, sigma1,50)
sample2 = np.random.multivariate_normal(mu2, sigma2,50)

data = np.concatenate((sample1,sample2))
#initialize cluster assignmnet at random
cluster = np.random.choice([0,1],replace = True, size = 100)

plt.scatter(data[:,0],data[:,1], c = cluster)

#update centroid step

def update_centroids(data,cluster):
    x1 = (data[:,0]*cluster).sum()/cluster.sum()
    y1 = (data[:,1]*cluster).sum()/cluster.sum()
    
    x2 = (data[:,0]*(1-cluster)).sum()/(1-cluster).sum()
    y2 = (data[:,1]*(1-cluster)).sum()/(1-cluster).sum()
    
    return([x1,y1],[x2,y2])
    
#Assign points step
    
def assign_cluster(data,centroid1,centroid2):
    d1 = np.sqrt((data[:,0]-centroid1[0])**2+(data[:,1]-centroid1[1])**2)
    d2 = np.sqrt((data[:,0]-centroid2[0])**2+(data[:,1]-centroid2[1])**2)
    
    d = np.greater(d1,d2).astype(int)
    
    return(d)

    
    
    
count = 0

while count < 20:
    
    centroid1, centroid2 = update_centroids(data,cluster)
    cluster = assign_cluster(data,centroid1,centroid2)
    count = count + 1
    
    
    
plt.scatter(data[:,0],data[:,1], c = cluster)  
    
    
    