import numpy as np
import matplotlib.pyplot as plt
import initial_condition as ic
import brownian as bw
import potential as pt
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

R=0.75

fstar=1.

nuAb=2e-3
nuAd=1e-3
nuA=1e-3

nuBb=2e-3
nuBd=1e-3
nuB=1e-3

DA=1e-1
DB=1e-1

NA=100
NB=100
mu=1.

# Parameters for the scheme

L=3


# [x,y]=np.meshgrid(np.arange(-L+dx/2.,L,dx),np.arange(-L+dy/2.,L,dy))
# X=np.zeros((2,M))
# X[0,:]=np.reshape(x,(M))
# X[1,:]=np.reshape(y,(M))

dt=0.01
T=1.
Nt=int(T/dt)


# Initial condition

[XA0,XB0]=ic.ini_uniform(L,NA,NB)


XA=XA0
XB=XB0

XAmem=[XA0]
XBmem=[XB0]

# Time scheme

for i in range(Nt):
    time_start=time.clock()
    
    # second membre
    # for A
    argsAA=(KAA, nuAAc, nuAAd, R)
    argsAB=(KAB, nuABc, nuABd, R)
    BA=bw.brownian(DA,NA)
    WA=pt.potential(NA,NB,argsAA,argsAB,XA,XB,i,mu,R)
    
    # for B
    argsBB=(KBB, nuBBc, nuBBd, R)
    argsBA=(KBA, nuBAc, nuBAd, R)
    BB=bw.brownian(DB,NB)
    WB=pt.potential(NB,NA,argsBB,argsBA,XB,XA,i,mu,R)
    
  
    XAnew=XA+dt*(WA+BA)
    XBnew=XB+dt*(WB+BB)
    
    
    
    # updating
    XA=XAnew
    XB=XBnew
    
    XAmem.append(XA)
    XBmem.append(XB)

# 
# plt.plot(XA0[0,:],XA0[1,:],"o")
# plt.plot(XB0[0,:],XB0[1,:],"or")
# plt.colorbar()
#plt.show()

fig = plt.figure()
t = np.arange(0., T,dt)
#axis = fig.add_subplot(111, xlim=(-L, L), ylim=(-L, L))
pp,ppp = plt.plot([],[],[],[])
plt.xlim(-L,L)
plt.ylim(-L,L)

def update(iframe):
    global pp, ppp
    # pp.set_data([XAmem[iframe][0],XBmem[iframe][0]],[XAmem[iframe][1],XBmem[iframe][1]])
    # plt.setp(pp,marker="o",linewidth=0)
    pp.set_data(XAmem[iframe][0],XAmem[iframe][1])
    plt.setp(pp,marker="o",color="b",linewidth=0)
    
    ppp.set_data(XBmem[iframe][0],XBmem[iframe][1])
    plt.setp(ppp,marker="o",color="r",linewidth=0)
    
    return pp,ppp
    
    
anim = animation.FuncAnimation(fig, update, frames=t.size, interval=1, repeat=True)
plt.show()









   