# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 21:41:12 2019

@author: Mdaneshvar
"""
import numpy as np


def sigmoid(z):
    sigmoid = 1/(1+np.exp(-z))
    return sigmoid


def sigmoid_Derivative(z):
    sigmoid_Derivative = np.exp(-z)/(1+np.exp(-z))**2
    return sigmoid_Derivative

def cost_function(w, Y, X, b):
    m = np.shape(X)[1]
    n = np.shape(X)[0]
    J = np.ones(m)
    X = np.asarray(X)
    Y = np.asarray(Y)
    w = np.asarray(w)
    yhat = sigmoid(np.dot(w.T,X)+b)  #1 by m
    yhat_recip = np.reciprocal(yhat)
    yhat_J_recip = np.reciprocal(J-yhat)
    log_yhat = np.log(yhat)
    log_yhat_1 = np.log(J-yhat)
    cost = (-1/m)*(np.dot(Y,log_yhat) + np.dot((J-Y),log_yhat_1))
    dyhat = (-1/m)*(np.multiply(Y,yhat_recip) - np.multiply((J-Y),yhat_J_recip))  # 1 by m
    dyhat_dw = np.zeros((m,n))
    for i in range (0,m) :
        dyhat_dw[i,:] = sigmoid_Derivative(np.dot(w,X[:,i])+b[i])*np.transpose(X)[i,:]  #m by n
    dw = np.dot(dyhat,dyhat_dw)
    
    return [yhat, log_yhat, log_yhat_1, yhat_recip,
            yhat_J_recip, dyhat, dyhat_dw, dw, cost]

X = [[1,1],[-1,0],[0,0],[0,0.1]]
Y = [0,1]
b = [0,0]

alpha = [0.001,0.001,0.001,0.001]
v = [0,0,0,0]
# Grdaient descent method
for i in range(0,1000):
    yhat, log_yhat, log_yhat_1, yhat_recip, yhat_J_recip, dyhat, dyhat_dw, dw, cost = cost_function(v,Y,X,b)
    v = v - np.multiply(alpha,np.transpose(dw))
    print(dw)    
   
    

