import numpy as np

def birthdeath(X,beta,delta):
    birth=np.random.binomial(1,beta,size=(X.shape[1]))
    death=np.random.binomial(1,delta,size=(X.shape[1]))
    Xbirth=X[:,birth==1]
    Y=X
    if any(death):
        # print(death==1)
        Y=np.delete(Y,np.where(death==1),axis=1)
    if any(birth):
        # print(birth==1)
        Y=np.concatenate((Y,Xbirth),axis=1)
    return Y
    
# beta=0.5
# delta=0.5
# X=np.array([np.arange(0,10,1),np.arange(0,10,1)])
# 
# Y=birthdeath(X,beta,delta)
