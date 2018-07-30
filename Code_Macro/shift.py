import numpy as np
import scipy.sparse as sp
import time

def shift_E(u,Nx,Ny):
    # decale les inconnues de la grille d'un cran vers la droite
    A=np.reshape(u,(Nx,Ny))
    Aini=np.reshape(A[:,Nx-1],(Ny,1))
    Afin=A[:,0:Nx-1]
    As=np.concatenate((Afin,Aini),axis=1)
    s=np.reshape(As,(Nx*Ny))
    return s

def shift_W(u,Nx,Ny):
    # decale les inconnues de la grille d'un cran vers la gauche
    A=np.reshape(u,(Nx,Ny))
    Aini=np.reshape(A[:,0],(Ny,1))
    Afin=A[:,1:Nx]
    As=np.concatenate((Afin,Aini),axis=1)
    s=np.reshape(As,(Nx*Ny))
    return s

def shift_S(u,Nx,Ny):
    # decale les inconnues de la grille d'un cran vers le bas
    A=np.reshape(u,(Nx,Ny))
    Aini=np.reshape(A[0,:],(1,Nx))
    Afin=A[1:Ny,:]
    As=np.concatenate((Afin,Aini),axis=0)
    s=np.reshape(As,(Nx*Ny))
    return s

def shift_N(u,Nx,Ny):
    # decale les inconnues de la grille d'un cran vers le haut
    A=np.reshape(u,(Nx,Ny))
    Aini=u[0:Ny-1,:]
    Afin=np.reshape(A[Ny-1,:],(1,Nx))
    As=np.concatenate((Afin,Aini),axis=0)
    s=np.reshape(As,(Nx*Ny))
    return s
    
# Nx=Ny=10
# u=np.arange(Nx*Ny)
# 
# time_start=time.clock()
# s=shift_S(u,Nx,Ny)
# print(time.clock()-time_start)
# time_start=time.clock()
# 
# A=np.reshape(u,(Nx,Ny))
# Aini=np.reshape(A[0,:],(1,Nx))
# Afin=A[1:Nx,:]
# As=np.concatenate((Afin,Aini),axis=0)
# ss=np.reshape(As,(Nx*Ny))
# print(time.clock()-time_start)
# 
# print(s-ss)
















































