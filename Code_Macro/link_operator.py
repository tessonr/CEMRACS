import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as sl
import convol as cv

def link_operator(dx,dy,Nx,Ny,f,g,args1,args2,PHI1,PHI2):
    Xi=cv.discrete_convol(dx,dy,PHI1,PHI2,X,f,g,args1,args2)
    
    [FluxE,FluxW,FluxN,FluxS]=flux(Xi,fE,fW,fS,fN,Nx,Ny)
    LO=-(FluE-FluxW)/dx-(FluxN-FluxS)/dy
    return L0

def flux(Xi,fE,fW,fS,fN,Nx,Ny):
    # Definition des differents flux utilises dans l'operateur de lien
    uE=-(shift_W(Xi)-Xi)/dx
    FluxE=(np.abs(uE)+uE)/2.*fE+(np.abs(uE)-uE)/2.*shift_W(fW,Nx,Ny)
    uW=-(Xi-shift_E(Xi))/dx
    FluxW=(np.abs(uW)+uW)/2.*shift_E(fE,Nx,Ny)+(np.abs(uW)-uW)/2.*fW
    uN=-(shift_S(Xi)-Xi)/dy
    FluxN=(np.abs(uN)+uN)/2.*fN+(np.abs(uN)-uN)/2.*shift_S(fS,Nx,Ny)
    uS=-(Xi-shift_N(Xi))/dy
    FluxS=(np.abs(uS)+uS)/2.*shift_N(fN)+(np.abs(uS)-uS)/2.*fS
    return [FluxE,FluxW,FluxN,FluxS]
    
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

#S=shift_W(np.arange(0,20,1),4,5)
#S=shift_E(np.arange(0,20,1),4,5)
#S=shift_S(np.arange(0,10,1),2,5)
# S=shift_N(np.arange(0,10,1),2,5)