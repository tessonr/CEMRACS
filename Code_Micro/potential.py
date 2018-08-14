import numpy as np
import gradphiST as gph

def potential(NS,NT,args1,args2,XS,XT,i,mu,R,L):
    W=np.zeros((2,NS))
    V1=np.ones((1,XS.shape[1]))
    V2=np.ones((1,XT.shape[1]))
    Vx=np.array([[2*L],[0]])
    Vy=np.array([[0],[2*L]])
    for i in range(NS):
        x=XS[0,i]
        y=XS[1,i]
        YS=np.delete(np.matmul(np.reshape(XS[:,i],(2,1)),V1)-XS,i,axis=1)
        YT=np.matmul(np.reshape(XS[:,i],(2,1)),np.ones((1,XT.shape[1])))-XT
        normS=np.linalg.norm(YS,axis=0)
        normT=np.linalg.norm(YT,axis=0)
        W[:,i]=W[:,i]-mu/NS*np.sum(gph.gradphiST(YS[:,normS<R],*args1),axis=1)-mu/NT*np.sum(gph.gradphiST(YT[:,normT<R],*args2),axis=1)
        if (x<-L+R):
            YS=np.delete(np.matmul(np.reshape(XS[:,i],(2,1)),V1)-XS-Vx@V1,i,axis=1)
            YT=np.matmul(np.reshape(XS[:,i],(2,1)),np.ones((1,XT.shape[1])))-XT-np.matmul(Vx,V2)
            normS=np.linalg.norm(YS,axis=0)
            normT=np.linalg.norm(YT,axis=0)
            W[:,i]=W[:,i]-mu/NS*np.sum(gph.gradphiST(YS[:,normS<R],*args1),axis=1)-mu/NT*np.sum(gph.gradphiST(YT[:,normT<R],*args2),axis=1)
        if (x>L-R):
            YS=np.delete(np.matmul(np.reshape(XS[:,i],(2,1)),V1)-XS+np.matmul(Vx,V1),i,axis=1)
            YT=np.matmul(np.reshape(XS[:,i],(2,1)),np.ones((1,XT.shape[1])))-XT+np.matmul(Vx,V2)
            normS=np.linalg.norm(YS,axis=0)
            normT=np.linalg.norm(YT,axis=0)
            W[:,i]=W[:,i]-mu/NS*np.sum(gph.gradphiST(YS[:,normS<R],*args1),axis=1)-mu/NT*np.sum(gph.gradphiST(YT[:,normT<R],*args2),axis=1)
        if (y<-L+R):
            YS=np.delete(np.matmul(np.reshape(XS[:,i],(2,1)),V1)-XS-np.matmul(Vy,V1),i,axis=1)
            YT=np.matmul(np.reshape(XS[:,i],(2,1)),np.ones((1,XT.shape[1])))-XT-np.matmul(Vy,V2)
            normS=np.linalg.norm(YS,axis=0)
            normT=np.linalg.norm(YT,axis=0)
            W[:,i]=W[:,i]-mu/NS*np.sum(gph.gradphiST(YS[:,normS<R],*args1),axis=1)-mu/NT*np.sum(gph.gradphiST(YT[:,normT<R],*args2),axis=1)
        if (y>L-R):
            YS=np.delete(np.matmul(np.reshape(XS[:,i],(2,1)),V1)-XS+np.matmul(Vy,V1),i,axis=1)
            YT=np.matmul(np.reshape(XS[:,i],(2,1)),np.ones((1,XT.shape[1])))-XT+np.matmul(Vy,V2)
            normS=np.linalg.norm(YS,axis=0)
            normT=np.linalg.norm(YT,axis=0)
            W[:,i]=W[:,i]-mu/NS*np.sum(gph.gradphiST(YS[:,normS<R],*args1),axis=1)-mu/NT*np.sum(gph.gradphiST(YT[:,normT<R],*args2),axis=1)
    return W

# X = np.array([np.arange(0.,2.,1), np.zeros(2)])
# Y=np.array([[0],[1]])
# NX=2
# NY=1
# KST=1.
# nuSTc=1.
# nuSTd=1.
# R=2.
# L=10.
# i=0
# mu=1.
# W=potential(NX,NY,(KST, nuSTc, nuSTd, R),(KST, nuSTc, nuSTd, R),X,Y,i,mu,R,L)


