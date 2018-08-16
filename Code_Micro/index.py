import numpy as np

def upindex(X,R,L):
    N=X.shape[1]
    Ind=locindex(X,R,L)
    M=int(2*L/R)**2
    firstindex=np.zeros((M)).astype(int)
    vectcell=np.zeros((N)).astype(int)
    for i in range(N):
        if (np.floor(firstindex[Ind[i]])==0):
            firstindex[Ind[i]]=i
        else:
            j=int(firstindex[Ind[i]])
            while (np.floor(vectcell[j])!=0):
                j=vectcell[j]
            vectcell[j]=i
    return [firstindex,vectcell]


def locindex(X,R,L):
    N=int(2*L/R)
    Index=np.floor((X[0,:]+L)/R)+np.floor((X[1,:]+L)/R)*N
    Index=Index.astype(int)
    return Index
    
    