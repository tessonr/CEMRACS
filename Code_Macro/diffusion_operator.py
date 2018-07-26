import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

def diffusion(Nx,Ny,dx,dy,D):
    # Size of matrix
    N=Nx*Ny
    # Definitions of Diagonals
    d0=(2/dx**2+2/dy**2)*np.ones(N)
    d1=(-1/dx**2)*np.ones(N)
    d2=(-1/dy**2)*np.ones(N)
    # Definitions of diffusion matrix
    A=sp.lil_matrix(sp.spdiags([d0,d1,d1,d2,d2],[0,1,-1,Nx,-Nx],N,N))
    return A
    
# Nx=Ny=5
# dx=dy=1
# D=1
# 
# A=diffusion(Nx,Ny,dx,dy,D)