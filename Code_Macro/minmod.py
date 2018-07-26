import numpy as np
import matplotlib.pyplot as plt


def minmod(a,b,c):
    if a>0 and b>0 and  c>0:
        z=min(a,b,c)
    elif a<0 and b<0 and  c<0:
        z=max(a,b,c)
    else: 
        z=0
        
    return z
            
        

        
    
    
    
    