import numpy as np

def birthdeath(X,beta,delta,r):
    birth=np.random.binomial(1,beta)
    death=np.random.binomial(1,delta)
    al=np.random.uniform(0,2*np.pi,np.sum(birth==1))
    Xbirth=X[:,birth==1]+r*np.array([np.cos(al),np.sin(al)])
    Y=X
    if any(death):
        # print(death==1)
        Y=np.delete(Y,np.where(death==1),axis=1)
    if any(birth):
        # print(birth==1)
        Y=np.concatenate((Y,Xbirth),axis=1)
    return Y
    
def bdrate(XS,XT,R0,b0,d0,thS,Nstar,L):
    betaS=np.zeros(XS.shape[1])
    deltaS=np.zeros(XS.shape[1])
    
    V1=np.ones((1,XS.shape[1]))
    V2=np.ones((1,XT.shape[1]))
    Vx=np.array([[2*L],[0]])
    Vy=np.array([[0],[2*L]])
    for i in range(XS.shape[1]):
        x=XS[0,i]
        y=XS[1,i]
        
        YS=np.reshape(XS[:,i],(2,1))@V1-XS
        YT=np.reshape(XS[:,i],(2,1))@np.ones((1,XT.shape[1]))-XT
        normS=np.linalg.norm(YS,axis=0)
        normT=np.linalg.norm(YT,axis=0)
    
        Npop=np.sum(normS<R0)+np.sum(normT<R0)
        if (x<-L+R0):
            YS=np.reshape(XS[:,i],(2,1))@V1-XS-Vx@V1
            YT=np.reshape(XS[:,i],(2,1))@np.ones((1,XT.shape[1]))-XT-Vx@V2
            normS=np.linalg.norm(YS,axis=0)
            normT=np.linalg.norm(YT,axis=0)
            Npop=Npop+np.sum(normS<R0)+np.sum(normT<R0)
        if (x>L-R0):
            YS=np.reshape(XS[:,i],(2,1))@V1-XS+Vx@V1
            YT=np.reshape(XS[:,i],(2,1))@np.ones((1,XT.shape[1]))-XT+Vx@V2
            normS=np.linalg.norm(YS,axis=0)
            normT=np.linalg.norm(YT,axis=0)
            Npop=Npop+np.sum(normS<R0)+np.sum(normT<R0)
        if (y<-L+R0):
            YS=np.reshape(XS[:,i],(2,1))@V1-XS-Vy@V1
            YT=np.reshape(XS[:,i],(2,1))@np.ones((1,XT.shape[1]))-XT-Vy@V2
            normS=np.linalg.norm(YS,axis=0)
            normT=np.linalg.norm(YT,axis=0)
            Npop=Npop+np.sum(normS<R0)+np.sum(normT<R0)
        if (y>L-R0):
            YS=np.reshape(XS[:,i],(2,1))@V1-XS+Vy@V1
            YT=np.reshape(XS[:,i],(2,1))*np.ones((1,XT.shape[1]))-XT+Vy@V2
            normS=np.linalg.norm(YS,axis=0)
            normT=np.linalg.norm(YT,axis=0)
            Npop=Npop+np.sum(normS<R0)+np.sum(normT<R0)
        
        betaS[i]=b0-(b0-thS)*Npop/Nstar
        deltaS[i]=d0+(thS-d0)*Npop/Nstar
    return [betaS,deltaS]
    
beta=0.5
delta=0.4
XS=np.array([np.arange(0,10,1),np.arange(0,10,1)])
XT=np.concatenate((np.reshape(np.arange(0,5,0.5),(1,10)),np.reshape(np.arange(1,6,0.5),(1,10))))

R0=2.
th=0.45
Nstar=4
L=5
r=0.1

[b,d]=bdrate(XS,XT,R0,beta,delta,th,Nstar,L)
XSnew=birthdeath(XS,b,d,r)
