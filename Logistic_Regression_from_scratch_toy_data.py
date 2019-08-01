# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:53:46 2019

@author: gakel
"""
import numpy as np
from scipy.optimize import minimize

X = np.array([1,1,2,4,8,6,8]).reshape(7,1)
Y = np.array([0,0,0,0,1,1,1]).reshape(7,1)


#initialize weight vector
theta = np.array([.1,-.2])

#add column of ones to X
o = np.ones([7,1])
X = np.concatenate((o,X),1)


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


h(b,X)




