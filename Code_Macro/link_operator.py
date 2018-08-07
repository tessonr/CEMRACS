import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import convol as cv
import convol_fft as cvf
import linear_interpol as li
import shift as sft
import time
import phiST as ph

def link_operator_fft(dx,dy,Nx,Ny,th,f,g,phifft1,phifft2):
    # compute the discretization of the link operator
    # args1 is the parameters for function PHI1
    # args2 is the parameters for function PHI2
    # time_start0=time.clock()
    # time_start=time.clock()
    Xi=cvf.discrete_convol_fft(dx, dy, phifft1, phifft2, f, g, Nx, Ny)
    # print(time.clock()-time_start)
    # time_start=time.clock()
    fE=li.interp_E(f,th,dx,Nx,Ny)
    fW=li.interp_W(f,th,dx,Nx,Ny)
    fS=li.interp_S(f,th,dx,Nx,Ny)
    fN=li.interp_N(f,th,dx,Nx,Ny)
    # print(time.clock()-time_start)
    # time_start=time.clock()

    [FluxE,FluxW,FluxN,FluxS]=flux(Xi,fE,fW,fS,fN,dx,dy,Nx,Ny)
    LO=-(FluxE-FluxW)/dx-(FluxN-FluxS)/dy
    # print(time.clock()-time_start0)
    return LO
    
def link_operator(dx,dy,Nx,Ny,X,th,f,g,args1,args2,PHI1,PHI2):
    # compute the discretization of the link operator
    # args1 is the parameters for function PHI1
    # args2 is the parameters for function PHI2
    # time_start0=time.clock()
    # time_start=time.clock()
    Xi=cv.discrete_convol(dx,dy,PHI1,PHI2,X,f,g,args1,args2)
    # print(time.clock()-time_start)
    # time_start=time.clock()
    fE=li.interp_E(f,th,dx,Nx,Ny)
    fW=li.interp_W(f,th,dx,Nx,Ny)
    fS=li.interp_S(f,th,dx,Nx,Ny)
    fN=li.interp_N(f,th,dx,Nx,Ny)
    # print(time.clock()-time_start)
    # time_start=time.clock()
    
    [FluxE,FluxW,FluxN,FluxS]=flux(Xi,fE,fW,fS,fN,dx,dy,Nx,Ny)
    LO=-(FluxE-FluxW)/dx-(FluxN-FluxS)/dy
    # print(time.clock()-time_start0)
    return LO

# flux
#def flux(Xi,fE,fW,fS,fN,dx,dy,Nx,Ny):
#    # Definition des differents flux utilises dans l'operateur de lien
#    uE=-(sft.shift_W(Xi,Nx,Ny)-Xi)/dx
#    FluxE=(np.abs(uE)+uE)/2.*fE+(np.abs(uE)-uE)/2.*sft.shift_W(fW,Nx,Ny)
#    uW=-(Xi-sft.shift_E(Xi,Nx,Ny))/dx
#    FluxW=(np.abs(uW)+uW)/2.*sft.shift_E(fE,Nx,Ny)+(np.abs(uW)-uW)/2.*fW
#    uN=-(sft.shift_S(Xi,Nx,Ny)-Xi)/dy
#    FluxN=(np.abs(uN)+uN)/2.*fN+(np.abs(uN)-uN)/2.*sft.shift_S(fS,Nx,Ny)
#    uS=-(Xi-sft.shift_N(Xi,Nx,Ny))/dy
#    FluxS=(np.abs(uS)+uS)/2.*sft.shift_N(fN,Nx,Ny)+(np.abs(uS)-uS)/2.*fS
#    return [FluxE,FluxW,FluxN,FluxS]
    


# flux debug test with changed sign 
def flux(Xi,fE,fW,fS,fN,dx,dy,Nx,Ny):
    # Definition des differents flux utilises dans l'operateur de lien
    uE=-(sft.shift_W(Xi,Nx,Ny)-Xi)/dx
    FluxE=((np.abs(uE)+uE)/2.)*fE-((np.abs(uE)-uE)/2.)*sft.shift_W(fW,Nx,Ny)
    uW=-(Xi-sft.shift_E(Xi,Nx,Ny))/dx
    FluxW=((np.abs(uW)+uW)/2.)*sft.shift_E(fE,Nx,Ny)-((np.abs(uW)-uW)/2.)*fW
    uN=-(sft.shift_S(Xi,Nx,Ny)-Xi)/dy
    FluxN=((np.abs(uN)+uN)/2.)*fN-((np.abs(uN)-uN)/2.)*sft.shift_S(fS,Nx,Ny)
    uS=-(Xi-sft.shift_N(Xi,Nx,Ny))/dy
    FluxS=((np.abs(uS)+uS)/2.)*sft.shift_N(fN,Nx,Ny)-((np.abs(uS)-uS)/2.)*fS
    
    M=np.amax([(np.abs(uE)+uE)/2.,(np.abs(uE)-uE)/2., (np.abs(uW)+uW)/2.,(np.abs(uW)-uW)/2., 
             (np.abs(uN)+uN)/2.,(np.abs(uN)-uN)/2.,(np.abs(uS)+uS)/2.,(np.abs(uS)-uS)/2.])
    CFL=dx/(4*M)
    print("CFL: ",CFL)
    return [FluxE,FluxW,FluxN,FluxS]






#S=shift_W(np.arange(0,20,1),4,5)
#S=shift_E(np.arange(0,20,1),4,5)
#S=shift_S(np.arange(0,10,1),2,5)
# S=shift_N(np.arange(0,10,1),2,5)

# L=7.5
# dx=0.3
# dy=0.3
# Nx=int(2*L/dx)
# Ny=int(2*L/dy)
# M=Nx*Ny
# 
# [x,y]=np.meshgrid(np.arange(-L+dx/2.,L,dx),np.arange(-L+dy/2.,L,dy))
# X=np.zeros((2,M))
# X[0,:]=np.reshape(x,(M))
# X[1,:]=np.reshape(y,(M))
# 
# th=2.
# 
# f=g=np.ones(M)
# args1=args2=(1.,1.,1.,1.)
# 
# L0=link_operator(dx,dy,Nx,Ny,X,th,f,g,args1,args2,ph.phiST,ph.phiST)




