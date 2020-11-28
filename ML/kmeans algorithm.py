# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:26:08 2019

@author: Mdaneshvar
"""
import numpy as np
import random


def cluster(X, new_sample, num_clus):
    
    s_clus=random.choices(X, k = num_clus)


    m = np.shape(X)[0] 
 

    for num_iter in range(100):

        # cluster ssignment
        ind = []    
        for i in range(m):

            b = []

            for j in range (num_clus):
                a = np.dot((X[i]-s_clus[j]),(X[i]-s_clus[j]))
                b.append(a)
            ind.append(np.argmin(b))

        # moving clusters
        d = [] 
        for j in range(num_clus):
            c = []
            for i in range(m):

                if ind[i] == j:

                    c.append(X[i])

            d.append(np.mean(c))
            s_clus[j] = d[j]
            
    # Predict the new sample        
    a_new = []
    for j in range(num_clus):
        a_new.append(np.dot((new_sample-s_clus[j]),(new_sample-s_clus[j])))
        new_ind = np.argmin(a_new) 
   
    return [ind, s_clus, new_ind]


X = np.array([[4,5],[-2,-3],[3,4],[-5,4],[-4,-3],[4,6],[3,-2],[-3,3]])
new_sample = [4,3]  
num_clus = 2

[ind, s_clus, new_ind] = cluster(X, new_sample, num_clus)  
print('Samples = ', X)
print('Class of samples X = ', ind)
print('Location of clusters = ', s_clus)
print('New sample is =', new_sample)
print('Class of new sample is = ', new_ind)
