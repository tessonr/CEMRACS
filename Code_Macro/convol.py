import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import phiST as ph

def evaluate_potential(PHI,X,*args): # evaluation de phi en les points de la grille
    N=X.shape[1]
    P=np.zeros((N,N))
    for i in range(N):
        P[i,:]=PHI(np.repeat(np.array([X[:,i]]).T,N,axis=1)-X,*args)
    return P

def convol(dx,dy,phi1,phi2,f,g): # calcul d'une convolution
    Xi=dx*dy*(phi1.T @ f+ phi2.T @ g)
    return Xi
    
def discrete_convol(dx,dy,PHI1,PHI2,X,f,g,args1,args2): # calcul du terme de convolution discrete global
    phi1=evaluate_potential(PHI1,X,*args1)
    phi2=evaluate_potential(PHI2,X,*args2)
    Xi=convol(dx,dy,phi1,phi2,f,g)
    return Xi

# def F(f,x,*args):
#     print(*args)
#     z=f(x,*args)
#     return z
#     
# def phi(x,a,b):
#     z=a*x+b
#     return z
#     
# def G(f,x,arg1,arg2):
#     z1=F(f,x,*arg1)
#     z2=F(f,x,*arg2)
#     return [z1,z2]
#     
# z=G(phi,1,(1,1),(1,1))
# print(z)
    
# dx=dy=1
# 
# KST, nuSTc, nuSTd, R = 1., 1., 1., 2.
# x1 = np.array([[0,1,2,0,1,2,0,1,2],[0,0,0,1,1,1,2,2,2]])
# # f=g=0.5*np.ones((4,1))
# 
# f=g=np.reshape(np.arange(0,9,1),(9,1))
# 
# Xi=discrete_convol(dx,dy,ph.phiST,ph.phiST,x1,f,g,(KST, nuSTc, nuSTd, R),(KST, nuSTc, nuSTd, R ))
# 
# # Resultats theorique : (12-4*np.srqt(2))/2
    

