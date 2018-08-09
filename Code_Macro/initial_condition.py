import numpy as np

def random_square(N,pert,dx,dy):
    f=np.ones(N)+pert*np.random.uniform(0,1,N)
    f=f/(np.sum(f)*dx*dy)
    return f
    
N=10
dx=1
dy=1
pert=0.01

f=random_square(N,pert,dx,dy)