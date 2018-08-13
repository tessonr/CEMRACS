import numpy as np
import matplotlib.pyplot as plt

# Model parameters 

KAA=2.
KAB=1.
KBA=2.
KBB=2.
K=[KAA,KAB,KBA,KBB]

nuAAc=1.
nuABc=1.
nuBAc=1.
nuBBc=1.
nuc=[nuAAc,nuABc,nuBAc,nuBBc]

nuAAd=1.
nuABd=1.
nuBAd=1.
nuBBd=1.
nud=[nuAAd,nuABd,nuBAd,nuBBd]

R=1.

fstar=1.

nuAb=2e-3
nuBb=2e-3
nub=[nuAb,nuBb]

DA=1e-4
DB=1e-4
D=[DA,DB]

# Computation of the function s* with and without logistic

def cprimeST(fS,KST,nuSTc,nuSTd,R):
    c=(2*np.pi*KST*fS*nuSTc*R**4)/nuSTd
    return c

def sstarL(f,K,nuc,nub,D,R):
    fA,fB=f[0],f[1]
    KAA,KAB,KBA,KBB=K[0],K[1],K[2],K[3]
    nuAAc,nuABc,nuBAc,nuBBc=nuc[0],nuc[1],nuc[2],nuc[3]
    nuAb,nuBb=nub[0],nub[1]
    DA,DB=D[0],D[1]
    cAA=cprimeST(fA,KAA,nuAAc,nuAAc,R)
    cAB=cprimeST(fA,KAB,nuABc,nuABc,R)
    cBA=cprimeST(fB,KBA,nuBAc,nuBAc,R)
    cBB=cprimeST(fB,KBB,nuBBc,nuBBc,R)
    
    s=((24*DA+cAA)*nuBb*fB+(24*DB+cBB)*nuAb*fA)/(nuBb*fB*cAB+nuAb*fA*cBA)
    return s

def sstarC(f,K,nuc,D,R):
    fA,fB=f[0],f[1]
    KAA,KAB,KBA,KBB=K[0],K[1],K[2],K[3]
    nuAAc,nuABc,nuBAc,nuBBc=nuc[0],nuc[1],nuc[2],nuc[3]
    DA,DB=D[0],D[1]
    cAA=cprimeST(fA,KAA,nuAAc,nuAAc,R)
    cAB=cprimeST(fA,KAB,nuABc,nuABc,R)
    cBA=cprimeST(fB,KBA,nuBAc,nuBAc,R)
    cBB=cprimeST(fB,KBB,nuBBc,nuBBc,R)
    
    s=np.sqrt((24*DA+cAA)*(24*DB+cBB)/(cAB*cBA))
    return s
    
# Plot of the functions sstar

ds=0.01
ff=np.arange(ds,fstar,ds)
f=[ff,fstar-ff]
sL=sstarL(f,K,nuc,nub,D,R)
sC=sstarC(f,K,nuc,D,R)

plt.figure(0)
plt.plot(ff,sL)
plt.plot(ff,sC)
plt.legend(["s* with logistic","s* without logistic"])
plt.xlabel("fA*")
plt.title('Case III')
plt.show()

# Computation of the minimum value of s*(fA,fB) with logistic

def fkST(KST,nuSTc,nuSTd,R):
    c=(2*np.pi*KST*nuSTc*R**4)/nuSTd
    return c

def fSLminus(DA,DB,nuAb,nuBb,fstar):
    if DA*nuBb-nuAb*DB==0:
        fS=fstar/2
    else:
        fS=(DA*nuAb*fstar-fstar*np.sqrt(DA*DB*nuAb*nuBb))/(DA*nuBb-nuAb*DB)
    return fS
    
def fSCminus(KAA,KBB,nuAAc,nuBBc,R,DA,DB,fstar):
    if DA*nuBb-nuAb*DB==0:
        fS=fstar/2
    else:
        k3=fkST(KAA,nuAAc,nuAAd,R)
        k4=fkST(KBB,nuBBc,nuBBd,R)
        detilde=(24*DA*DB+DA*k4*fstar)**2+24*DB**2*DA*k3*fstar-24*DA**2*DB*k4*fstar+DA*DB*k3*k4*fstar**2-DA**2*k4**2*fstar**2
        fS=(-(24*DA*DB+DA*k4*fstar)+np.sqrt(detilde))/(DB*k3-DA*k4)
    return fS

fAminL=fSLminus(DA,DB,nuAb,nuBb,fstar)
print("fAminL :",fAminL)
print("fBminL :",fstar-fAminL)
fminL=[fAminL,fstar-fAminL]
sLmin=sstarL(fminL,K,nuc,nub,D,R)
print("sLmin",sLmin)

fAminC=fSCminus(KAA,KBB,nuAAc,nuBBc,R,DA,DB,fstar)
print("fAminC :",fAminC)
print("fBminC :",fstar-fAminC)
fminC=[fAminC,fstar-fAminC]
sCmin=sstarC(fminC,K,nuc,D,R)
print("sCmin",sCmin)
