import numpy as np
import scipy.sparse as sp
import phiST as ph

def convol_fft(dx, dy, fftphi, f, Nx, Ny):
    # shifting the Fourier transforms
    # shift_fftphi = np.fft.fftshift(fftphi)
    # shift_fftf = np.fft.fftshift(fftf)
    # fftphi = np.reshape(fftphi,(Nx,Ny))
    fftf = np.fft.fft2(np.reshape(f,(Ny,Nx)))
    fftresult = fftphi * fftf
    ifftresult = dx*dy*np.real(np.fft.ifft2(fftresult))
    result = np.reshape(ifftresult,(Nx*Ny,))
    return result


def discrete_convol_fft(dx,dy,fftphi1, fftphi2, f, g, Nx, Ny):  # calcul d'une convolution
    Xi = convol_fft(dx, dy, fftphi1, f, Nx, Ny) + convol_fft(dx, dy, fftphi2, g, Nx, Ny)
    return Xi


# def discrete_convol(dx, dy, PHI1, PHI2, X, f, g, args1, args2):  # calcul du terme de convolution discrete global
#     phi1 = evaluate_potential(PHI1, X, *args1)
#     phi2 = evaluate_potential(PHI2, X, *args2)
#     Xi = convol(dx, dy, phi1, phi2, f, g)
#     return Xi



# dx=dy=1
# 
# KST, nuSTc, nuSTd, R = 1., 1., 1., 2.
# Nx, Ny = 3, 3
# x1 = np.array([[0,1,2,0,1,2,0,1,2],[0,0,0,1,1,1,2,2,2]])
# f=g=np.reshape(np.arange(0.,9.,1.),(Nx,Ny))
# # print(ph.phiST(x1,(KST, nuSTc, nuSTd, R)))
# # print(ph.phiST(x1,(KST, nuSTc, nuSTd, R)).shape)
# testphi1 = np.fft.fft2(np.reshape(ph.phiST(x1,(KST, nuSTc, nuSTd, R)),(Nx,Ny)))
# testphi2 = np.fft.fft2(np.reshape(ph.phiST(x1,(KST, nuSTc, nuSTd, R)),(Nx,Ny)))
# Xi=discrete_convol_fft(dx, dy, testphi1, testphi2, f, g, Nx, Ny)
# 
# # print('computed value: ', [10.745166, 13.0588745, 19.40202025, 17.6862915, 20., 26.34314575, 36.71572875, 39.02943725, 45.372583])
# # print('exact value: ', (12-4*np.sqrt(2))/2)
# # print('computed value: ', Xi)
# # Resultats theorique : (12-4*np.srqt(2))/2