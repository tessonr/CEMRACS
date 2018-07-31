import numpy as np

def brownian(D,N):
    B=np.sqrt(2.*D)*np.random.normal(size=(2,N))
    return B