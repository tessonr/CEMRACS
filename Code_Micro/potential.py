import numpy as np
import gradphiST as gph
import index as ind

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
            YS=np.delete(np.matmul(np.reshape(XS[:,i],(2,1)),V1)-XS-np.matmul(Vx,V1),i,axis=1)
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
    
    
def potential_index(NS,NT,args1,args2,XS,XT,firstindexS,vectcellS,firstindexT,vectcellT,i,mu,R,L):
    W=np.zeros((2,NS))
    IndS=ind.locindex(XS,R,L)
    IndT=ind.locindex(XT,R,L)
    M=int(2*L/R)
    for i in range(NS):
        # Cellule dans la même case
        j=IndS[i]
        k=firstindexS[j]
        if k!=i:
            W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        while vectcellS[k]!=0:
            k=vectcellS[k]
            if k!=i:
                W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        k=firstindexT[j]
        W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        while vectcellT[k]!=0:
            k=vectcellT[k]
            W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        # Cellule dans la case W
        if j%M==0:
            k=firstindexS[j+M-1]
        else:
            k=firstindexS[j-1]
        W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        while vectcellS[k]!=0:
            k=vectcellS[k]
            W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        k=firstindexT[j]
        W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        while vectcellT[k]!=0:
            k=vectcellT[k]
            W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        # Cellule dans la case SW
        if j==0:
            k=firstindexS[M**2-1]
        elif j%M==0:
            k=firstindexS[j-1]
        elif j<M:
            k=firstindexS[M**2-M+j]
        else:
            k=firstindexS[j-M-1]
        W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        while vectcellS[k]!=0:
            k=vectcellS[k]
            W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        k=firstindexT[j]
        W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        while vectcellT[k]!=0:
            k=vectcellT[k]
            W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        # Cellule dans la case S
        if j<M:
            k=firstindexS[M**2-M+j]
        else:
            k=firstindexS[j-M]
        W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        while vectcellS[k]!=0:
            k=vectcellS[k]
            W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        k=firstindexT[j]
        W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        while vectcellT[k]!=0:
            k=vectcellT[k]
            W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        # Cellule dans la case SE
        if j==M-1:
            k=firstindexS[M**2-M]
        elif j%M==M-1:
            k=firstindexS[j+1-2*M]
        elif j<M:
            k=firstindexS[M**2-M+j]
        else:
            k=firstindexS[j-M+1]
        W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        while vectcellS[k]!=0:
            k=vectcellS[k]
            W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        k=firstindexT[j]
        W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        while vectcellT[k]!=0:
            k=vectcellT[k]
            W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        # Cellule dans la case E
        if j%M==M-1:
            k=firstindexS[j-M+1]
        else:
            k=firstindexS[j+1]
        W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        while vectcellS[k]!=0:
            k=vectcellS[k]
            W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        k=firstindexT[j]
        W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        while vectcellT[k]!=0:
            k=vectcellT[k]
            W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        # Cellule dans la case NE
        if j==M**2-1:
            k=firstindexS[0]
        elif j%M==M-1:
            k=firstindexS[j+1]
        elif j>M**2-M-1:
            k=firstindexS[j-M**2+M]
        else:
            k=firstindexS[j+M+1]
        W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        while vectcellS[k]!=0:
            k=vectcellS[k]
            W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        k=firstindexT[j]
        W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        while vectcellT[k]!=0:
            k=vectcellT[k]
            W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        # Cellule dans la case N
        if j>M**2-M-1:
            k=firstindexS[j-M**2+M]
        else:
            k=firstindexS[j+M]
        W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        while vectcellS[k]!=0:
            k=vectcellS[k]
            W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        k=firstindexT[j]
        W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        while vectcellT[k]!=0:
            k=vectcellT[k]
            W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        # Cellule dans la case NW
        if j==M**2-M:
            k=firstindexS[M-1]
        elif j%M==0:
            k=firstindexS[j+2*M-1]
        elif j>M**2-M-1:
            k=firstindexS[j-M**2+M]
        else:
            k=firstindexS[j+M-1]
        W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        while vectcellS[k]!=0:
            k=vectcellS[k]
            W[:,i]=W[:,i]-mu/NS*gph.gradphiST(XS[:,i]-XS[:,k],*args1)
        k=firstindexT[j]
        W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
        while vectcellT[k]!=0:
            k=vectcellT[k]
            W[:,i]=W[:,i]-mu/NT*gph.gradphiST(XS[:,i]-XT[:,k],*args2)
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


