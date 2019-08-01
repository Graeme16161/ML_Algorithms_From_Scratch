# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:48:10 2019

@author: gakel
"""

#import packages
import numpy as np
from sklearn.datasets import load_iris
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


#load iris data set
iris = load_iris(True)
X = iris[0]
Y = iris[1]

# take only two species for binary classification
X = X[0:100,:]
Y = Y[0:100]

#initialize weight vector
theta = np.array([0,0,0,0,0])

#add column of ones to X
o = np.ones([100,1])
X = np.concatenate((o,X),1)

#hypothesis function, returns probability that classified as 1
#theta -> vector of weights
#x -> observation
def h(theta,x):
    
    z = np.dot(x,theta)
    
    s = 1/(1+np.exp(-1*z))
    
    return(s)
    
    

#cost function
def J(theta,X,Y):
    
    n = len(Y)
    
    probability = h(theta,X)
    
    cost1 = (1-Y).T*np.log(1-probability)
    cost2 = (-1*Y).T*np.log(probability)
    
    cost = cost2-cost1
    
    return(cost.sum()/n)
    
    

#minimie J
m = minimize(J,x0 = theta, args = (X,Y))

#resulting coefficients
b = m.x
    
print("The minimization was successful: %s " % m.success)
print("b0: %.2f" % b[0])
print("b1: %.2f" % b[1])
print("b2: %.2f" % b[2])
print("b3: %.2f" % b[3])
print("b4: %.2f" % b[4])
    


