import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import convol as cv
import diffusion_operator as do
import logistic_operator as lo
import link_operator as lop
import phiST as ph
import initial_condition as ic
import time
import matplotlib.animation as animation
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

nuAb=2e-3
nuAd=1e-3
nuA=1e-3

nuBb=2e-3
nuBd=1e-3
nuB=1e-3

DA=1e-4
DB=1e-4

# Parameters for the scheme

L=7.5
dx=0.3
dy=0.3
Nx=int(2*L/dx)
Ny=int(2*L/dy)
M=Nx*Ny

[x,y]=np.meshgrid(np.arange(-L+dx/2.,L,dx),np.arange(-L+dy/2.,L,dy))
X=np.zeros((2,M))
X[0,:]=np.reshape(x,(M))
X[1,:]=np.reshape(y,(M))

dt=5e-3
T=1.
Nt=int(T/dt)

th=2.

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

#construction of the intra and inter species interaction potentials their FOurier Transform

#for A
argsAA=(KAA, nuAAc, nuAAd, R)
argsAB=(KAB, nuABc, nuABd, R)
phiAA = ph.phiST(X,argsAA)
phiAB = ph.phiST(X,argsAB)
#Fast Fourier Transform of phiAA and phiAB
rphiAA = np.reshape(phiAA,(Nx,Ny))
rphiAB = np.reshape(phiAB,(Nx,Ny))
fftphiAA = np.fft.fft2(rphiAA)
fftphiAB = np.fft.fft2(rphiAB)

#for B
argsBB=(KBB, nuBBc, nuBBd, R)
argsBA=(KBA, nuBAc, nuBAd, R)
phiBA = ph.phiST(X,argsBA)
phiBB = ph.phiST(X,argsBB)
#Fast Fourier Transform of phiBA and phiBB
rphiBA = np.reshape(phiBA,(Nx,Ny))
rphiBB = np.reshape(phiBB,(Nx,Ny))
fftphiBA = np.fft.fft2(rphiAA)
fftphiBB = np.fft.fft2(rphiAB)
# Time scheme

for i in range(Nt):
    print(i)
    time_start=time.clock()
    
    # second membre
    # for A
    # argsAA=(KAA, nuAAc, nuAAd, R)
    # argsAB=(KAB, nuABc, nuABd, R)
    # LO_A=dt*lop.link_operator(dx,dy,Nx,Ny,X,th,fA,fB,argsAA,argsAB,ph.phiST,ph.phiST)
    LO_A=dt*lop.link_operator_fft(dx,dy,Nx,Ny,th,fA,fB,fftphiAA,fftphiAB)

    FR_A=dt*lo.logistic(fA,fB,fstar,nuA)
    
    vecA=fA+LO_A+FR_A
    
    # for B
    # argsBB=(KBB, nuBBc, nuBBd, R)
    # argsBA=(KBA, nuBAc, nuBAd, R)
    # LO_B=dt*lop.link_operator(dx,dy,Nx,Ny,X,th,fB,fA,argsBB,argsBA,ph.phiST,ph.phiST)
    LO_B=dt*lop.link_operator_fft(dx,dy,Nx,Ny,th,fA,fB,fftphiBA,fftphiBB)

    FR_B=dt*lo.logistic(fB,fA,fstar,nuB)
    
    vecB=fB+LO_B+FR_B
    
    print(time.clock()-time_start)
    
    # solving of the system
    time_start=time.clock()
    
    fnewA=sla.spsolve(MatA,vecA)
    fnewB=sla.spsolve(MatB,vecB)
    
    print(time.clock()-time_start)
    
    # updating
    fA=fnewA
    fB=fnewB
    
    FA[:,i+1]=fA
    FB[:,i+1]=fB

# fig = plt.figure(3)
# plt.contourf(x,y,np.reshape(fA,(Nx,Ny)),cmap=plt.cm.hot)
# plt.colorbar()
# plt.show()

t = np.arange(0., T, dt)
fig = plt.figure(3)

cmap = plt.get_cmap("Spectral")
def update(iframe):
    plt.clf()
    fig.canvas.draw()
    plt.subplot(121)
    plt.imshow(np.reshape(FA[:, iframe], (Nx, Ny)), interpolation='spline16', cmap=cmap)
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(np.reshape(FB[:, iframe], (Nx, Ny)),interpolation='spline16', cmap=cmap)
    plt.colorbar()


anim = animation.FuncAnimation(fig, update, frames=t.size, interval=1, repeat=True)
plt.show()
   
   
   