# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 08:32:50 2019

@author: gakel
"""

#import pacakges
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#get data
data = load_iris()
data = data.data

#split between target variable and predictors
Y = data[:,0]
X = data[:,1:4]


#get params function
def get_b_values(a,X,Y):
    size = X.shape[1]
    
    X1 = np.dot(X.T,X)
    X2 = X1 + a*np.eye(size)
    X3 = np.dot(np.linalg.inv(X2),X.T)
    
    b = np.dot(X3,Y)
    
    return(b)
    
a_values = np.linspace(0,1000,100)

b_array = []
mse = []

for a in a_values:
    
    b = get_b_values(a,X,Y)
    b_array.append(b)


plotting_array = pd.DataFrame(b_array)
plotting_array = plotting_array.rename(columns={0: "b1", 1: "b2", 2:"b3"})

plotting_array.plot()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(plotting_array.b1, 'r', plotting_array.b2, 'g', plotting_array.b3, 'b')
ax.axhline(y=0, color='black', linestyle='--')
ax.set_xlabel("Lambda")
ax.set_ylabel("Beta Estimate")
ax.set_title("Ridge Regression Trace", fontsize=16)
ax.legend(labels=['b0','b1','b2'])
ax.grid(True)
