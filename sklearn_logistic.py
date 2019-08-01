# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:08:34 2019

@author: gakel
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


iris = load_iris(True)
X = iris[0]
Y = iris[1]

# take only two species for binary classification
X = X[0:100,:]
Y = Y[0:100]

clf = LogisticRegression()

fit = clf.fit(X,Y)

fit.coef_
fit.intercept_



X1 = X[:,:2]


clf1 = LogisticRegression()

fit1 = clf.fit(X1,Y)

fit1.coef_
fit1.intercept_

v_theta = np.concatenate((fit1.coef_[0],fit1.intercept_))


t_theta1 = np.array([2.5,-4,-1])
t_theta2 = np.array([3,-4,-1])