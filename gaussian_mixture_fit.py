# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:09:14 2019

@author: gakel
"""

#odeint

import numpy as np
from statistics import variance
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt


#Simulate data
mu1 = 8
sigma1 = 1
mu2 = 1
sigma2 = 2

g1 = np.random.normal(mu1,sigma1,50)
g2 = np.random.normal(mu2,sigma2,30)

data = np.concatenate((g1,g2))

#plt.hist(data, bins = 20)



#initialize parameters
#phi is the wieght of a component/gaussian 

mu_hats = np.random.choice(data,size = 2, replace = False)
mu_hat1 = mu_hats[0]
mu_hat2 = mu_hats[1]

sigma_hat1 = sigma_hat2 = variance(data)

weight = .5

#expectation function
#returns vector of gammas which are the responsibility of gaussian 1 for x_i
#
#note, code is written for 2 guassians

def expectation_step(x,weight, mu_hat1,mu_hat2,sigma_hat1,sigma_hat2):
    num = weight * norm(mu_hat2,sqrt(sigma_hat2)).pdf(x)
    denom = (1-weight)*norm(mu_hat1,sqrt(sigma_hat1)).pdf(x) + weight * norm(mu_hat2,sqrt(sigma_hat2)).pdf(x)
    
    return(num/denom)
    

#maximization function
#returns updated mus, sigmas and weight
    
def maximization_step(x,gamma):
    m1 = ((1-gamma)*x).sum()/(1-gamma).sum()
    m2 = ((gamma)*x).sum()/(gamma).sum()
    
    s1 = ((1-gamma)*((x - m1)**2)).sum()/(1-gamma).sum()
    s2 = ((gamma)*((x - m2)**2)).sum()/(gamma).sum()
    
    w = gamma.sum()/len(gamma)
    
    return(m1,m2,s1,s2,w)


#iterate untill covergence

dif = 1
count = 0
while(dif > .00001 and count < 100):
    
    gamma = expectation_step(data, weight, mu_hat1,mu_hat2, sigma_hat1,sigma_hat2)
    mu_hat1,mu_hat2, sigma_hat1,sigma_hat2,new_weight = maximization_step(data,gamma)
    
    dif = abs(new_weight - weight)
    weight = new_weight
    count = count +1
    

    




print("Final Estimated Parameters:")
print("Iterations: %d" % count)
print("Gaussian 1: N(%.2f,%.2f)" % (mu_hat1,sigma_hat1))
print("Gaussian 2: N(%.2f,%.2f)" % (mu_hat2,sigma_hat2))


