import numpy as np
import matplotlib.pyplot as plt


def minmod(a,b,c):
    M=[a,b,c]
    P=np.zeros(tuple(a.shape))
    P[np.amin(M,axis=0)>0]=np.amin(M,axis=0)[np.amin(M,axis=0)>0]
    P[np.amax(M,axis=0)<0]=np.amax(M,axis=0)[np.amax(M,axis=0)<0]
    return P
    
    # if a>0 and b>0 and  c>0:
    #     z=min(a,b,c)
    # elif a<0 and b<0 and  c<0:
    #     z=max(a,b,c)
    # else: 
    #     z=0
    #     
    # return z
    
    
# a=-np.arange(0,5,1)
# b=np.arange(2,-3,-1)
# c=-np.ones(5)
# 
# print(minmod(a,b,c))