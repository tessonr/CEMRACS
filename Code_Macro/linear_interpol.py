import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import shift as sft
import minmod as mmd
import time

def interp_E(f,th,dx,Nx,Ny):
    time_start=time.clock()
    slope=(sft.shift_W(f,Nx,Ny)-sft.shift_E(f,Nx,Ny))/(2*dx)
    fE=f+slope*dx/2
    print(time.clock()-time_start)
    time_start=time.clock()
    fE[fE<0]=f[fE<0]+(dx/2.)*mmd.minmod(th*(f[fE<0]-sft.shift_E(f,Nx,Ny)[fE<0])/dx,slope[fE<0],th*(sft.shift_W(f,Nx,Ny)[fE<0]-f[fE<0])/dx)
    print(time.clock()-time_start)
    time_start=time.clock()
    return fE
    
def interp_W(f,th,dx,Nx,Ny):
    slope=(sft.shift_W(f,Nx,Ny)-sft.shift_E(f,Nx,Ny))/(2*dx)
    fW=f-slope*dx/2
    fW[fW<0]=f[fW<0]-(dx/2.)*mmd.minmod(th*(f[fW<0]-sft.shift_E(f,Nx,Ny)[fW<0])/dx,slope[fW<0],th*(sft.shift_W(f,Nx,Ny)[fW<0]-f[fW<0])/dx)
    return fW
    
def interp_N(f,th,dy,Nx,Ny):
    slope=(sft.shift_S(f,Nx,Ny)-sft.shift_N(f,Nx,Ny))/(2*dy)
    fN=f+slope*dy/2
    fN[fN<0]=f[fN<0]+(dy/2.)*mmd.minmod(th*(f[fN<0]-sft.shift_N(f,Nx,Ny)[fN<0])/dy,slope[fN<0],th*(sft.shift_S(f,Nx,Ny)[fN<0]-f[fN<0])/dy)
    return fN
    
def interp_S(f,th,dy,Nx,Ny):
    slope=(sft.shift_S(f,Nx,Ny)-sft.shift_N(f,Nx,Ny))/(2*dy)
    fS=f-slope*dy/2
    fS[fS<0]=f[fS<0]-(dy/2.)*mmd.minmod(th*(f[fS<0]-sft.shift_N(f,Nx,Ny)[fS<0])/dy,slope[fS<0],th*(sft.shift_S(f,Nx,Ny)[fS<0]-f[fS<0])/dy)
    return fS
    
# f=np.arange(0,9,1)
# dx=1
# th=1
# Nx=Ny=3
# 
# print(interp_E(f,th,dx,Nx,Ny))