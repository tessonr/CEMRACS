import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

def diffusion(Nx,Ny,dx,dy):
    # Size of matrix
    N=Nx*Ny
    # Definitions of Diagonals
    d0x=(2/dx**2)*np.ones(N)
    d0y=(2/dy**2)*np.ones(N)
    d1=(-1/dx**2)*np.ones(N)
    d2=(-1/dy**2)*np.ones(N)
    # Definitions of diffusion matrix
    Al=sp.lil_matrix(sp.spdiags([d0x,d1,d1],[0,1,-1],Nx,Nx))
    Al[Nx-1,0]=-1/dx**2
    Al[0,Nx-1]=-1/dx**2
    Ah=sp.lil_matrix(sp.block_diag(tuple([Al]*Ny)))
    Av=sp.lil_matrix(sp.spdiags([d0y,d2,d2,d2,d2],[0,Nx,-Nx,N-Nx,Nx-N],N,N))
    A=Ah+Av
    return A
    
# Nx=Ny=3
# dx=dy=1
# D=1
# 
# A=diffusion(Nx,Ny,dx,dy)