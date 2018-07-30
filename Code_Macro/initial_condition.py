import numpy as np

def random_square(N,dx,dy):
    f=np.ones(N)+np.random.uniform(0,1,N)
    f=f/(np.sum(f)*dx*dy)
    return f
    
N=10
dx=1
dy=1

f=random_square(N,dx,dy)