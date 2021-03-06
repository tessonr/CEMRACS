import numpy as np
import scipy.sparse as sp
import shift as sft
import minmod as mmd
import time

def interp_EW(f,th,dx,Nx,Ny):
    # computation of the slopes (centered)
    slope=(sft.shift_W(f,Nx,Ny)-sft.shift_E(f,Nx,Ny))/(2*dx)
    # computation of linear interpolations
    fE=f+slope*dx/2.
    fW=f-slope*dx/2.
    # change the negative interpolations
    BadI=(fE<0)|(fW<0)
    newslopes=mmd.minmod(th*(f[BadI]-sft.shift_E(f,Nx,Ny)[BadI])/dx,slope[BadI],th*(sft.shift_W(f,Nx,Ny)[BadI]-f[BadI])/dx)
    fE[BadI]=f[BadI]+(dx/2.)*newslopes
    fW[BadI]=f[BadI]-(dx/2.)*newslopes
    return [fE,fW]

def interp_NS(f,th,dy,Nx,Ny):
    # computation of the slopes (centered)
    slope=(sft.shift_S(f,Nx,Ny)-sft.shift_N(f,Nx,Ny))/(2*dy)
    # computation of linear interpolations
    fN=f+slope*dy/2.
    fS=f-slope*dy/2.
    # change the negative interpolations
    BadI=(fN<0)|(fS<0)
    newslopes=mmd.minmod(th*(f[BadI]-sft.shift_N(f,Nx,Ny)[BadI])/dy,slope[BadI],th*(sft.shift_S(f,Nx,Ny)[BadI]-f[BadI])/dy)
    fN[BadI]=f[BadI]+(dy/2.)*newslopes
    fS[BadI]=f[BadI]-(dy/2.)*newslopes
    return [fN,fS]



# def interp_E(f,th,dx,Nx,Ny):
#     slope=(sft.shift_W(f,Nx,Ny)-sft.shift_E(f,Nx,Ny))/(2*dx)
#     fE=f+slope*dx/2.
#     fE[fE<0]=f[fE<0]+(dx/2.)*mmd.minmod(th*(f[fE<0]-sft.shift_E(f,Nx,Ny)[fE<0])/dx,slope[fE<0],th*(sft.shift_W(f,Nx,Ny)[fE<0]-f[fE<0])/dx)
#     return fE
#     
# def interp_W(f,th,dx,Nx,Ny):
#     slope=(sft.shift_W(f,Nx,Ny)-sft.shift_E(f,Nx,Ny))/(2*dx)
#     fW=f-slope*dx/2.
#     fW[fW<0]=f[fW<0]-(dx/2.)*mmd.minmod(th*(f[fW<0]-sft.shift_E(f,Nx,Ny)[fW<0])/dx,slope[fW<0],th*(sft.shift_W(f,Nx,Ny)[fW<0]-f[fW<0])/dx)
#     return fW
#     
# def interp_N(f,th,dy,Nx,Ny):
#     slope=(sft.shift_S(f,Nx,Ny)-sft.shift_N(f,Nx,Ny))/(2*dy)
#     fN=f+slope*dy/2.
#     fN[fN<0]=f[fN<0]+(dy/2.)*mmd.minmod(th*(f[fN<0]-sft.shift_N(f,Nx,Ny)[fN<0])/dy,slope[fN<0],th*(sft.shift_S(f,Nx,Ny)[fN<0]-f[fN<0])/dy)
#     return fN
#     
# def interp_S(f,th,dy,Nx,Ny):
#     slope=(sft.shift_S(f,Nx,Ny)-sft.shift_N(f,Nx,Ny))/(2*dy)
#     fS=f-slope*dy/2.
#     fS[fS<0]=f[fS<0]-(dy/2.)*mmd.minmod(th*(f[fS<0]-sft.shift_N(f,Nx,Ny)[fS<0])/dy,slope[fS<0],th*(sft.shift_S(f,Nx,Ny)[fS<0]-f[fS<0])/dy)
#     return fS
    
# f=np.arange(0,9,1)
# dx=1
# th=1
# Nx=Ny=3
# 
# print(interp_EW(f,th,dx,Nx,Ny))