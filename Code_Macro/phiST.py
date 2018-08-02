import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


def phiST(x, args):
# def phiST(x, KST, nuSTc, nuSTd, R):
    """
        Return the potential energy for the particles.

        Parameters
        ----------
        KST : is the coefficient expressing the steepness of the spring
            real number
        nuSTc : link creation rate
            real number
        nuSTc : link destruction rate
            real number
        x : particles positions
            (2,$N_A$) matrix, where $N_A$ is the number of particles
            of type A for example

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> x1 = np.array([np.arange(0., 2., 0.01), np.arange(0., 2., 0.01)])
        >>> mod_x = np.sqrt((x1[0])**2 + (x1[1])**2)
        >>> KST, nuSTc, nuSTd, R = 0.2, 0.6, 0.3, np.sqrt(2)
        >>> f = phiST(x1,KST,nuSTc,nuSTd,R)
        >>> print(f)
        >>> plt.plot(mod_x,f)
        >>> plt.show()
        """
    # compute the module of x
    mod_x = np.sqrt((x[0,:])**2 + (x[1,:])**2)
    KST, nuSTc, nuSTd, R = args[0], args[1], args[2], args[3]
    # if mod_x <= R:
    f = ((nuSTc/nuSTd)*(KST/2)*(mod_x - R)**2)*(mod_x <= R) + 0.*(mod_x > R)
    # else f = 0
    return f


# Test the function

# x1 = np.array([np.arange(0., 2., 0.01), np.arange(0., 2., 0.01)])
# mod_x = np.sqrt((x1[0])**2 + (x1[1])**2)
# KST = 0.2
# nuSTc = 0.6
# nuSTd = 0.3
# R = np.sqrt(2)
# f = phiST(KST,nuSTc,nuSTd,x1,R)
# plt.plot(mod_x,f)
# plt.show()
