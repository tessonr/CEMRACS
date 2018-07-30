import numpy as np
import scipy.sparse as sp

def shift_E(u,Nx,Ny):
    # decale les inconnues de la grille d'un cran vers la droite
    A=np.diag(np.ones((Nx-1)),-1)
    A[0,Nx-1]=1
    P=sp.block_diag(tuple([A]*Ny))
    s=P@u
    return s

def shift_W(u,Nx,Ny):
    # decale les inconnues de la grille d'un cran vers la gauche
    A=np.diag(np.ones((Nx-1)),1)
    A[Nx-1,0]=1
    P=sp.block_diag(tuple([A]*Ny))
    s=P@u
    return s

def shift_S(u,Nx,Ny):
    # decale les inconnues de la grille d'un cran vers le bas
    A=np.diag(np.ones((Nx*Ny-Nx)),Nx)+np.diag(np.ones((Nx)),-Nx*Ny+Nx)
    s=A@u
    return s

def shift_N(u,Nx,Ny):
    # decale les inconnues de la grille d'un cran vers le haut
    A=np.diag(np.ones((Nx*Ny-Nx)),-Nx)+np.diag(np.ones((Nx)),Nx*Ny-Nx)
    s=A@u
    return s