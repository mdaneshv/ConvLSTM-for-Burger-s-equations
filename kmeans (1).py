# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:26:08 2019

@author: Mdaneshvar
"""
import numpy as np
import random



def cluster(X,num_clus):
    
    s_clus=random.choices(X,k=num_clus)
    return s_clus
 
    
# =============================================================================
# def cost_function(X,s):
#     
#     m=np.shape(X)[0]
#     
#     J=(1/m)*
# =============================================================================
    
X=[4,2,-3,4,10,-4,-1,5,-5,9,-5.3,4.3,-3.5,0.1] 
X=[10,14,12,7,9,0.1,-0.1,0.2,-0.3,0.01]
X=np.array([[4,5],[-2,-3],[3,4],[-5,4],[-4,-3],[4,6],[3,-2],[-3,3]])
num_clus=3
m=np.shape(X)[0] 
s_clus=cluster(X,num_clus) 

for num_iter in range(0,100):
    
    # cluster ssignment
    ind=[]    
    for i in range(m):
        
        b=[]
        
        for j in range (num_clus):
            a=np.dot((X[i]-s_clus[j]),(X[i]-s_clus[j]))
            b.append(a)
        ind.append(np.argmin(b) )
                    
    # moving clusters
    d=[] 
    for j in range(num_clus):
        c=[]
        for i in range(m):
            
            if ind[i]==j:
                
                c.append(X[i])
                
        d.append(np.mean(c))
        s_clus[j]=d[j]
print(ind)        
print(s_clus)   
             
      # prediction            
        
new_sample=[-7,6]
a_new=[]
for j in range(num_clus):
    a_new.append(np.dot((new_sample-s_clus[j]),(new_sample-s_clus[j])))
new_ind=np.argmin(a_new)   

print(new_ind) 
    
    
         
        
        
        
        
    
    
    
    
    
    