import numpy as np
import matplotlib.pyplot as plt

def logistic(f,g,fstar,nu):
    """
        Return the logisitc term for the birth and death of cells.

        Parameters
        ----------
        f : cell density of the first specie
            array of the corresponding densities
        g : cell density of the second specie
            array of the corresponding densities
        fstar : total capacity of the environement
            real number
        nu : growth rate of the first specie
            real number

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> x = np.arange(0,2,0.01)
        >>> f = np.ones(x.shape)*(x<=1)
        >>> g = 0.5*np.ones(x.shape)
        >>> fstar = 2
        >>> nu= 0.5
        >>> F = logistic(f,g,fstar,nu)
        >>> print(f)
        >>> plt.plot(x,F)
        >>> plt.show()
        """
        
    F=nu*f*(1-(f+g)/fstar)
    return F
    
# fstar=10
# nu=1
# f=np.ones((5,1))
# g=2*f
# 
# F=logistic(f,g,fstar,nu)