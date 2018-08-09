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
from matplotlib import colors

# Parameters for the model

KAA=2.
KAB=8.
KBA=8.
KBB=2.

nuAAc=1.
nuABc=1.
nuBAc=1.
nuBBc=1.

nuAAd=1.
nuABd=1.
nuBAd=1.
nuBBd=1.

R=1.

nuAb=2e-3
nuAd=1e-3
nuA= 0.0

nuBb=2e-3
nuBd=1e-3
nuB= 0.0

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

deltat=1e-2
tt=0.
tps=np.arange(0,1,1)
T=20.

# parameter for the CFL
pCFL=0.9

# Definition of the population protability
# Nstar = 20.
# R0 = 2.
# fstar=(Nstar)/(np.pi*R0**2)
fstar= 1e-2
th=1

# Initial condition

pert=0.01

fA0=ic.random_square(M,pert,dx,dy)
fB0=ic.random_square(M,pert,dx,dy)

fA=fA0
fB=fB0

FA=np.reshape(fA0,(M,1))
FB=np.reshape(fB0,(M,1))

# matrix of the linear system
MatD=sp.csc_matrix(do.diffusion(Nx,Ny,dx,dy))
MatT=sp.csc_matrix(sp.lil_matrix(np.eye(M)))

#construction of the intra and inter species interaction potentials their FOurier Transform

#for A
argsAA=(KAA, nuAAc, nuAAd, R)
argsAB=(KAB, nuABc, nuABd, R)
phiAA = ph.phiST(X,argsAA)
phiAB = ph.phiST(X,argsAB)
#Fast Fourier Transform of phiAA and phiAB
rphiAA = np.reshape(phiAA,(Ny,Nx))
rphiAB = np.reshape(phiAB,(Ny,Nx))
fftphiAA = np.fft.fft2(rphiAA)
fftphiAB = np.fft.fft2(rphiAB)

#for B
argsBB=(KBB, nuBBc, nuBBd, R)
argsBA=(KBA, nuBAc, nuBAd, R)
phiBA = ph.phiST(X,argsBA)
phiBB = ph.phiST(X,argsBB)
#Fast Fourier Transform of phiBA and phiBB
rphiBA = np.reshape(phiBA,(Ny,Nx))
rphiBB = np.reshape(phiBB,(Ny,Nx))
fftphiBA = np.fft.fft2(rphiBA)
fftphiBB = np.fft.fft2(rphiBB)
# Time scheme

while (tt<T):
    print(tt)
    time_start=time.clock()
    
    # computation of link operators (without dt) and CFL
    [LO_A,CFLA]=lop.link_operator_fft(dx,dy,Nx,Ny,th,fA,fB,fftphiAA,fftphiAB)
    [LO_B,CFLB]=lop.link_operator_fft(dx,dy,Nx,Ny,th,fA,fB,fftphiBA,fftphiBB)
    
    # computation of the time step
    dt=min(deltat,pCFL*CFLA,pCFL*CFLB)
    #dt=deltat
    
    # computation of the matrix
    MatA=(MatT+dt*DA*MatD)
    MatB=(MatT+dt*DB*MatD)
    
    # second membre
    # for A
    # argsAA=(KAA, nuAAc, nuAAd, R)
    # argsAB=(KAB, nuABc, nuABd, R)
    # LO_A=dt*lop.link_operator(dx,dy,Nx,Ny,X,th,fA,fB,argsAA,argsAB,ph.phiST,ph.phiST)
    LO_A=dt*LO_A

    FR_A=dt*lo.logistic(fA,fB,fstar,nuA)
    
    vecA=fA+LO_A
    
    # for B
    # argsBB=(KBB, nuBBc, nuBBd, R)
    # argsBA=(KBA, nuBAc, nuBAd, R)
    # LO_B=dt*lop.link_operator(dx,dy,Nx,Ny,X,th,fB,fA,argsBB,argsBA,ph.phiST,ph.phiST)
    LO_B=dt*LO_B

    FR_B=dt*lo.logistic(fB,fA,fstar,nuB)
    
    vecB=fB+LO_B
    
    #print(time.clock()-time_start)
    
    # solving of the system
    
    fnewA=sla.spsolve(MatA,vecA)
    fnewB=sla.spsolve(MatB,vecB)
    
    #print(time.clock()-time_start)
    
    if (any(fnewA<0)):
        print('\x1b[6;30;41m' + 'Negative values: ' + '\x1b[0m')
        print(fnewA[fnewA<0])
        print(vecA[fnewA<0])
        plt.spy(sp.csc_matrix(np.reshape(1*(fnewA<0), (Nx,Ny))))
        break
    # updating
    fA=fnewA
    fB=fnewB
    
    FA=np.concatenate((FA,np.reshape(fA,(M,1))),axis=1)
    FB=np.concatenate((FB,np.reshape(fB,(M,1))),axis=1)
    
    tt=tt+dt
    tps = np.concatenate((tps,np.reshape(tt,(1))))

# fig = plt.figure(3)
# plt.contourf(x,y,np.reshape(fA,(Nx,Ny)),cmap=plt.cm.hot)
# plt.colorbar()
# plt.show()

print(tt)

tpsbis = 0
fig = plt.figure(3)
cmap = plt.get_cmap("jet")
# def update(iframe):
#     global tpsbis
#     plt.clf()
#     fig.canvas.draw()
#     plt.subplot(121)
#     plt.imshow(np.reshape(FA[:, iframe],  (Ny, Nx)), origin='lower', aspect='auto', extent=[-L, L, -L, L],
#                interpolation='spline16', cmap=cmap)
#     plt.title('$f^A$ at t= '+ str(round(tpsbis,2)));
#     plt.colorbar()
# 
#     plt.subplot(122)
#     plt.imshow(np.reshape(FB[:, iframe], (Ny, Nx)), origin='lower', aspect='auto', extent=[-L, L, -L, L],
#                interpolation='spline16', cmap=cmap)
#     plt.title('$f^B$ at t= ' + str(round(tpsbis,2)));
#     plt.colorbar()
# 
#     # plt.pause(1e-6);
# 
#     tpsbis += tps[iframe]
#     
# anim = animation.FuncAnimation(fig, update, frames=tps.size, interval=1, repeat=True)

plt.subplot(221)
plt.imshow(np.reshape(fA0,  (Ny, Nx)), origin='lower', aspect='auto', extent=[-L, L, -L, L],interpolation='spline16', cmap=cmap)
plt.title('$f^A$ at t= '+ str(0));
plt.colorbar()

plt.subplot(222)
plt.imshow(np.reshape(fA,  (Ny, Nx)), origin='lower', aspect='auto', extent=[-L, L, -L, L],interpolation='spline16', cmap=cmap)
plt.title('$f^A$ at t= '+ str(round(tt,2)));
plt.colorbar()

plt.subplot(223)
plt.imshow(np.reshape(fB0,  (Ny, Nx)), origin='lower', aspect='auto', extent=[-L, L, -L, L],interpolation='spline16', cmap=cmap)
plt.title('$f^B$ at t= '+ str(0));
plt.colorbar()

plt.subplot(224)
plt.imshow(np.reshape(fB,  (Ny, Nx)), origin='lower', aspect='auto', extent=[-L, L, -L, L],interpolation='spline16', cmap=cmap)
plt.title('$f^A$ at t= '+ str(round(tt,2)));
plt.colorbar()
plt.show()

# Plot like in the article with 2 colors

One=np.ones((M,1))
Zer=np.zeros((M,1))
f=Zer
f0=Zer
f[fA>fB]=One[fA>fB]
f0[fA0>fB0]=One[fA0>fB0]

plt.figure(4)
cmap = plt.get_cmap("RdYlGn")

plt.subplot(121)
plt.imshow(np.reshape(f0,  (Ny, Nx)), origin='lower', aspect='auto', extent=[-L, L, -L, L],interpolation='spline16', cmap=cmap)
plt.title('$f$ at t= '+ str(0));
plt.colorbar()

plt.subplot(122)
plt.imshow(np.reshape(f,  (Ny, Nx)), origin='lower', aspect='auto', extent=[-L, L, -L, L],interpolation='spline16', cmap=cmap)
plt.title('$f$ at t= '+ str(round(tt,2)));
plt.colorbar()

plt.show()