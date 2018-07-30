import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import convol as cv
import diffusion_operator as do
import logistic_operator as lo
import link_operator as lop
import phiST as ph
import initial_condition as ic

# Parameters for the model

KAA=1.
KAB=1.
KBA=1.
KBB=1.

nuAAc=1.
nuABc=1.
nuBAc=1.
nuBBc=1.

nuAAd=1.
nuABd=1.
nuBAd=1.
nuBBd=1.

R=1.

fstar=1.

nuAb=1.
nuAd=1.
nuA=1.

nuBb=1.
nuBd=1.
nuB=1.

DA=1e-4
DB=1e-4

# Parameters for the scheme

L=7.5
dx=0.3
dy=0.3
Nx=int(2*L/dx)
Ny=int(2*L/dy)
M=Nx*Ny

X=

dt=1.
T=1.
Nt=int(T/dt)

# Initial condition

fA0=ic.random_square(M,dx,dy)
fB0=ic.random_square(M,dx,dy)

fA=fA0
fB=fB0

FA=np.zeros((M,Nt+1))
FA[:,0]=fA0
FB=np.zeros((M,Nt+1))
FB[:,0]=fB0

# matrix of the linear system
MatD=do.diffusion(Nx,Ny,dx,dy)
MatT=sp.lil_matrix(np.eye(M))
  
MatA=(MatT+dt*DA*MatD)
MatA=sp.csc_matrix(MatA)
MatB=(MatT+dt*DB*MatD)
MatA=sp.csc_matrix(MatB)


# Time scheme

for i in range(Nt):
   
   # second membre
   # for A
   argsAA=(KAA, nuAAc, nuAAd, R)
   argsAB=(KAB, nuABc, nuABd, R)
   LO_A=dt*lop.link_operator(dx,dy,Nx,Ny,X,fA,fB,argsAA,argsAA,ph.phiST,ph.phiST)
   
   FR_A=dt*lo.logistic(fA,fB,fstar,nuA)
   
   vecA=fA+LO_A+FR_A
   
   # for B
   argsBB=(KBB, nuBBc, nuBBd, R)
   argsBA=(KBA, nuBAc, nuBAd, R)
   LO_B=dt*lop.link_operator(dx,dy,Nx,Ny,X,fB,fA,argsBB,argsBA,ph.phiST,ph.phiST)
   
   FR_B=dt*lo.logistic(fB,fA,fstar,nuB)
   
   vecB=fB+LO_B+FR_B
   
   # solving of the system
   fnewA=sp.linalg.spsolve(MatA,vecA)
   fnewB=sp.linalg.spsolve(MatB,vecB)
   
   # updating
   fA=fnewA
   fB=fnewB
   
   FA[:,i+1]=fA
   FB[:,i+1]=fB
   
   
   
   
   