import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


def gradphiST(x, KST, nuSTc, nuSTd, R):
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
        >>> f = gradphiST(x1,KST,nuSTc,nuSTd,R)
        >>> print(f)
        >>> plt.plot(mod_x,f[0,:])
        >>> plt.show()
        """
    # compute the module of x
    mod_x = np.sqrt((x[0])**2 + (x[1])**2)
    # if mod_x <= R:
    ep=1e-10
    f = ((nuSTc/nuSTd)*KST*(mod_x - R)*x/(mod_x+ep))*(mod_x <= R) + 0.*(mod_x > R)
    # else f = 0
    return f


# Test the function

# x1 = np.array([np.arange(0.01, 2., 0.01), np.arange(0.01, 2., 0.01)])
# mod_x = np.sqrt((x1[0])**2 + (x1[1])**2)
# KST = 0.2
# nuSTc = 0.6
# nuSTd = 0.3
# R = np.sqrt(2)
# f = gradphiST(x1,KST,nuSTc,nuSTd,R)
# plt.plot(mod_x,f[0,:])
# plt.show()
