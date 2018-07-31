import numpy as np

def ini_uniform(L,NA,NB):
    XA0=np.random.uniform(-L,L,(2, NA))
    XB0=np.random.uniform(-L,L,(2, NB))
    return [XA0,XB0]


def ini_half():
    XA0=np.random.uniform(0,L,(2, NA))
    XB0=np.random.uniform(-L,0,(2, NB))
    return [XA0,XB0]
    