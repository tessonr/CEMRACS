import numpy as np
import gradphiST as gph

def potential(NS,NT,args1,args2,XS,XT,i,mu):
    YS=np.delete(np.reshape(XS[:,i],(2,1))*np.ones((1,XS.shape[1]))-XS,i,axis=1)
    YT=np.reshape(XS[:,i],(2,1))*np.ones((1,XT.shape[1]))-XT
    W=-mu/NS*np.sum(gph.gradphiST(YS,*args1),axis=1)-mu/NT*np.sum(gph.gradphiST(YT,*args2),axis=1)
    return W

# X = np.array([np.arange(0.,2.,1), np.zeros(2)])
# Y=np.array([[0],[1]])
# NX=2
# NY=1
# KST=1.
# nuSTc=1.
# nuSTd=1.
# R=2.
# i=0
# mu=1.
# 
# W=potential(NX,NY,(KST, nuSTc, nuSTd, R),(KST, nuSTc, nuSTd, R),X,Y,i,mu)