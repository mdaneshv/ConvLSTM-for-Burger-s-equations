import numpy as np
import matplotlib.pyplot as plt
import random


def initial(X, Y, n_h):
    
    n = np.shape(X)[0]
    m = np.shape(X)[1]
    n_y = np.shape(Y)[0]
    w1 = np.zeros((n_h, n))
    w2 = np.zeros((n_y, n_h))
    b1 = np.zeros((n_h, m))
    b2 = np.zeros((n_y, m))
    return w1, w2, b1, b2


def FP(X, w1, w2, b1, b2):
    
    Z1 = np.dot(w1,X) + b1   # (n_h,m)
    A1 = np.tanh(Z1)   #(n_h,m)
    Z2 = np.dot(w2,A1) + b2   #(n_y,m)
    A2 = 1/(1 + np.exp(-Z2))   #(n_y,m)
    parameters = {"Z1":Z1, "Z2":Z2, "A2":A2, "A1":A1}
    return parameters


def cost(X, Y, A, w1,
         w2, b1, b2):
    
    m = int(np.shape(X)[0])
    log1 = np.multiply(Y, np.log(A))
    log2 = np.multiply(1-Y, np.log(1-A))
    loss = -(1/m) * (np.sum(log1) + np.sum(log2))
    return loss

    
def BP(X, Y, w1, w2, parameters):
     
    m = np.shape(X)[0]
    A1 = parameters["A1"]
    A2 = parameters["A2"]
    
    dZ_2 = A2-Y   #(n_y,m)
    dW_2 = (1/m)*np.dot(dZ_2, A1.T)   #(n_y,n_h)
    db_2 = (1/m)*np.sum(dZ_2, axis=1, keepdims=True)
    dZ_1 = np.multiply(np.dot(w2.T, dZ_2),(1 - np.power(A1, 2)))   #(n_h,m)
    dW_1 = (1/m)*np.dot(dZ_1, X.T)  #(n_h,n_x)
    db_1 = (1/m)*np.sum(dZ_1, axis=1, keepdims=True)
    return dW_1, dW_2, db_1, db_2

    
lr = 20
reg = 0
n_h = 100


X = np.array([[1, 2, -1, -0.1],
              [2, 1, -1, 0],
              [-1, 0.5, 0, -2]])
Y = np.array([[1, 1, 0, 0]])
m = np.shape(X)[1]
w1, w2, b1, b2 = initial(X, Y, n_h) 
l = [] 


for i in range(0,300):
   
    parameters = FP(X, w1, w2, b1, b2)
    A2 = parameters["A2"]
    loss = cost(X, Y, A2, w1,
                w2, b1, b2)
    l = np.append(l,loss)
    dW_1, dW_2, db_1, db_2 = BP(X, Y, w1, w2,
                                parameters)
    
    w1 = (1 - reg/m)*w1 - lr*dW_1
    w2 = (1 - reg/m)*w2 - lr*dW_2
    b1 = b1 - lr*db_1
    b2 = b2 - lr*db_2


X_test = X = np.array([[0, 1, -1, 0.1],
              [-2, -1, 0, 0],
              [1, -0.5, -1, 2]])

parameters = FP(X_test, w1, w2, b1, b2) 

A2 = parameters["A2"]

print('predict=', A2)

iteration = np.arange(0,300)
plt.plot(iteration,l)
plt.title('loss')
plt.show()
