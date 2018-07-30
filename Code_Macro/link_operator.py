import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import convol as cv
import linear_interpol as li

def link_operator(dx,dy,Nx,Ny,X,th,f,g,args1,args2,PHI1,PHI2):
    # compute the discretization of the link operator
    # args1 is the parameters for function PHI1
    # args2 is the parameters for function PHI2
    Xi=cv.discrete_convol(dx,dy,PHI1,PHI2,X,f,g,args1,args2)
    fE=li.interp_E(f,th,dx,Nx,Ny)
    fW=li.interp_W(f,th,dx,Nx,Ny)
    fS=li.interp_S(f,th,dx,Nx,Ny)
    fN=li.interp_N(f,th,dx,Nx,Ny)
    
    [FluxE,FluxW,FluxN,FluxS]=flux(Xi,fE,fW,fS,fN,dx,dy,Nx,Ny)
    LO=-(FluxE-FluxW)/dx-(FluxN-FluxS)/dy
    return LO

def flux(Xi,fE,fW,fS,fN,dx,dy,Nx,Ny):
    # Definition des differents flux utilises dans l'operateur de lien
    uE=-(shift_W(Xi,Nx,Ny)-Xi)/dx
    FluxE=(np.abs(uE)+uE)/2.*fE+(np.abs(uE)-uE)/2.*shift_W(fW,Nx,Ny)
    uW=-(Xi-shift_E(Xi,Nx,Ny))/dx
    FluxW=(np.abs(uW)+uW)/2.*shift_E(fE,Nx,Ny)+(np.abs(uW)-uW)/2.*fW
    uN=-(shift_S(Xi,Nx,Ny)-Xi)/dy
    FluxN=(np.abs(uN)+uN)/2.*fN+(np.abs(uN)-uN)/2.*shift_S(fS,Nx,Ny)
    uS=-(Xi-shift_N(Xi,Nx,Ny))/dy
    FluxS=(np.abs(uS)+uS)/2.*shift_N(fN,Nx,Ny)+(np.abs(uS)-uS)/2.*fS
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