import numpy as np
import matplotlib.pyplot as plt
import scipy

def logistic(f,g,fstar,nu):
    F=nu*f*(1-(f+g)/fstar)
    return F
    
# fstar=10
# nu=1
# f=np.ones((5,1))
# g=2*f
# 
# F=logistic(f,g,fstar,nu)