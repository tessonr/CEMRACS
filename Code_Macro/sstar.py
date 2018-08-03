import numpy as np

# Model parameters 

KAA=1.
KAB=1.
KBA=1.
KBB=1.

nuAAc=1.
nuABc=1.
nuAbc=1.
nuBbc=1.

nuAAd=1.
nuABd=1.
nuAbd=1.
nuBbd=1.

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



k1=(2*np.pi*KAA*nuAAc*R**4)/nuAAd
k2=(2*np.pi*KBB*nuBBc*R**4)/nuBbd
k3=(2*np.pi*KAB*nuABc*R**4)/nuABd
k4=(2*np.pi*KBA*nuBAc*R**4)/nuAbd

alpha=-24*DA*nuBb+k1*nuBb*fstar+24*DB*nuAb+k2*nuAb*fstar
beta=-k1*nuBb-k2*nuAb
gamma=24*DA*nuBb*fstar
delta=nuBb*fstar*k3+nuAb*k4*fstar
ee=-nuBb*k3-nuAb*k4

def fSminus(DA,DB,nuAb,nuBb,fstar):
    if DA*nuBb-nuAb*DB==0:
        fS=fstar/2
    else:
        fS=(DA*nuAb*fstar-fstar*np.sqrt(DA*DB*nuAb*nuBb))/(DA*nuBb-nuAb*DB)
    return fS

def funz(fS,alpha,beta,gamma,delta,ee):
    FS=(alpha*fS+beta*fS**2+gamma)/(delta*fS+ee*fS**2)
    return FS


fS=fSminus(DA,DB,nuAb,nuBb,fstar)
FS=funz(fS,alpha,beta,gamma,delta,ee)

print(FS)




