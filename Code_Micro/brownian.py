import numpy as np

def brownian(D):
    B=np.sqrt(2.*D)*np.random.normal(size=2)
    return B