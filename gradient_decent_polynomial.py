# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:47:09 2019

@author: gakel
"""

import numpy as np
import pandas as pd

order_0 = 4
order_1 = 2
order_2 = -5
order_3 = 0
order_4 = 1


learning_rate = .01

starting_point = 0

line = np.linspace(-3,3,1000)

def fun(o0,o1,o2,o3,o4,x):
    y = o4*x**4+o3*x**3+o2*x**2+o1*x+o0
    return(y)
    
    
t = fun(order_0,order_1,order_2,order_3,order_4,line)

plot_fun_df = pd.DataFrame({"x" : line, "y" : t})

plot_fun_df.plot(x = "x",y = "y")

    
def deriv(o0,o1,o2,o3,o4,x):
    
    y = 4*o4*x**3 + 3*o3*x**2 + 2*o2*x+ o1
    return(y)
    
delta = 1
x = 1

while abs(delta) > .001:
    d = deriv(order_0,order_1,order_2,order_3,order_4,x)
    x_new = x - d*learning_rate
    
    delta = x_new - x
    x= x_new
    
print("Minimum found at x= %.2f" % x)








