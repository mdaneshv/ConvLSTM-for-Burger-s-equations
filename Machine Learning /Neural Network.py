#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(123)


def initial(X, Y, n_h):
    
    (n,m) = np.shape(X)
    n_y = Y.shape[0]
    w1 = np.random.randn(n_h, n)*0.01
    w2 = np.random.randn(n_y, n_h)*0.01
    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))
    return [w1, w2, b1, b2]


def forward_propagation(X, w1, w2, b1, b2):
    
    Z1 = np.dot(w1, X) + b1   # (n_h,m)
    A1 = np.tanh(Z1)   #(n_h,m)
    Z2 = np.dot(w2, A1) + b2   #(n_y,m)
    A2 = 1/(1 + np.exp(-Z2))   #(n_y,m)
    parameters = {"Z1":Z1, "Z2":Z2, "A2":A2, "A1":A1}
    return parameters


def loss_function(X, Y, A, w1,
                  w2, b1, b2):
    
    m = int(np.shape(X)[0])
    log1 = np.multiply(Y, np.log(A))
    log2 = np.multiply(1-Y, np.log(1-A))
    loss = -(1/m) * (np.sum(log1) + np.sum(log2))
    return loss

    
def back_propagation(X, Y, w1, w2, parameters):
     
    m = np.shape(X)[0]
    A1 = parameters["A1"]
    A2 = parameters["A2"]
    
    dZ_2 = A2-Y   #(n_y,m)
    dW_2 = (1/m)*np.dot(dZ_2, A1.T)   #(n_y,n_h)
    db_2 = (1/m)*np.sum(dZ_2, axis=1, keepdims=True)
    dZ_1 = np.multiply(np.dot(w2.T, dZ_2), 1 - np.power(A1, 2))   #(n_h,m)
    dW_1 = (1/m)*np.dot(dZ_1, X.T)   #(n_h,n_x)
    db_1 = (1/m)*np.sum(dZ_1, axis=1, keepdims=True)
    return [dW_1, dW_2, db_1, db_2]


def grdient_descent(X, Y, n_h, lr,
                    reg, num_iter): 
    
    m = np.shape(X)[0]
    [w1, w2, b1, b2] = initial(X, Y, n_h) 
    l = [] 
    for i in range(num_iter):
        parameters = forward_propagation(X, w1, w2, b1, b2)
        A2 = parameters["A2"]
        loss = loss_function(X, Y, A2, w1,
                             w2, b1, b2)
        l.append(loss)
        [dW_1, dW_2, db_1, db_2] = back_propagation(X, Y, w1, w2,
                                                    parameters)

        w1 = (1 - reg/m)*w1 - lr*dW_1
        w2 = (1 - reg/m)*w2 - lr*dW_2
        b1 = b1 - lr*db_1
        b2 = b2 - lr*db_2
    return [w1, w2, b1, b2]     
