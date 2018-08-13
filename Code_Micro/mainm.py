import numpy as np
import matplotlib.pyplot as plt
import initial_condition as ic
import brownian as bw
import potential as pt
import birthdeath as bd
import time
import matplotlib.animation as animation

# Parameters for the model

KAA = 2.
KAB = 8.
KBA = 8.
KBB = 2.

nuAAc = 1.
nuABc = 1.
nuBAc = 1.
nuBBc = 1.

nuAAd = 1.
nuABd = 1.
nuBAd = 1.
nuBBd = 1.

R = 1.
R0 = 2.
r = 0.5

DA = 1e-4
DB = 1e-4

NA = 50
NB = 50
mu = 1.

# betaA = 1e-3
# deltaA = 7e-4
# betaB = 1e-3
# deltaB = 7e-4
#
b0A = 1e-3
d0A = 1e-5
b0B = 1e-3
d0B = 1e-5

thA = 8e-4
thB = 8e-4

# b0A = 0.
# d0A = 0.
# b0B = 0.
# d0B = 0.
# 
# thA = 0.
# thB = 0.

Nstar = 20

# Parameters for the scheme

L = 7.5

# [x,y]=np.meshgrid(np.arange(-L+dx/2.,L,dx),np.arange(-L+dy/2.,L,dy))
# X=np.zeros((2,M))
# X[0,:]=np.reshape(x,(M))
# X[1,:]=np.reshape(y,(M))

dt = 0.1
T = 100
Nt = int(T / dt)

# Initial condition

[XA0, XB0] = ic.ini_uniform(L, NA, NB)

XA = XA0
XB = XB0

XAmem = [XA0]
XBmem = [XB0]

# Time scheme

for i in range(Nt):
    print(i*dt)
    time_start = time.clock()

    # second membre
    # for A
    argsAA = (KAA, nuAAc, nuAAd, R)
    argsAB = (KAB, nuABc, nuABd, R)
    BA = bw.brownian(DA, NA)
    WA = pt.potential(NA, NB, argsAA, argsAB, XA, XB, i, mu, R, L)

    # for B
    argsBB = (KBB, nuBBc, nuBBd, R)
    argsBA = (KBA, nuBAc, nuBAd, R)
    BB = bw.brownian(DB, NB)
    WB = pt.potential(NB, NA, argsBB, argsBA, XB, XA, i, mu, R, L)

    XAnew = XA + dt * WA + np.sqrt(dt) * BA
    XBnew = XB + dt * WB + np.sqrt(dt) * BB

    # boundary conditions

    XAnew[0, XAnew[0, :] > L] = XAnew[0, XAnew[0, :] > L] - 2 * L
    XAnew[0, XAnew[0, :] < -L] = XAnew[0, XAnew[0, :] < -L] + 2 * L
    XAnew[1, XAnew[1, :] > L] = XAnew[1, XAnew[1, :] > L] - 2 * L
    XAnew[1, XAnew[1, :] < -L] = XAnew[1, XAnew[1, :] < -L] + 2 * L

    XBnew[0, XBnew[0, :] > L] = XBnew[0, XBnew[0, :] > L] - 2 * L
    XBnew[0, XBnew[0, :] < -L] = XBnew[0, XBnew[0, :] < -L] + 2 * L
    XBnew[1, XBnew[1, :] > L] = XBnew[1, XBnew[1, :] > L] - 2 * L
    XBnew[1, XBnew[1, :] < -L] = XBnew[1, XBnew[1, :] < -L] + 2 * L

    # birth and death

    [betaA, deltaA] = bd.bdrate(XAnew, XBnew, R0, b0A, d0A, thA, Nstar, L)
    [betaB, deltaB] = bd.bdrate(XBnew, XAnew, R0, b0B, d0B, thB, Nstar, L)

    XAnew = bd.birthdeath(XAnew, betaA, deltaA, r)
    XBnew = bd.birthdeath(XBnew, betaB, deltaB, r)

    # updating
    XA = XAnew
    XB = XBnew
    NA = XA.shape[1]
    NB = XB.shape[1]
    print(NA)
    print(NB)

    XAmem.append(XA)
    XBmem.append(XB)

#
# plt.plot(XA0[0,:],XA0[1,:],"o")
# plt.plot(XB0[0,:],XB0[1,:],"or")
# plt.colorbar()
# plt.show()


plt.figure(0)
pp=plt.plot(XA0[0], XA0[1])
plt.setp(pp, marker="o", markersize=20., color="b", linewidth=0)

ppp=plt.plot(XB0[0], XB0[1])
plt.setp(ppp, marker="o", markersize=20., color="r", linewidth=0)

plt.title('t= ' + str(0)+", NA="+str(50)+", NA="+str(50))
plt.show()

plt.figure(1)
pp=plt.plot(XA[0], XA[1])
plt.setp(pp, marker="o", markersize=20., color="b", linewidth=0)

ppp=plt.plot(XB[0], XB[1])
plt.setp(ppp, marker="o", markersize=20., color="r", linewidth=0)

plt.title('t= ' + str(T)+", NA="+str(NA)+", NA="+str(NA))
plt.show()


fig = plt.figure(2)
t = np.arange(0., T, dt)
# axis = fig.add_subplot(111, xlim=(-L, L), ylim=(-L, L))
pp, ppp = plt.plot([], [], [], [])
plt.xlim(-L + 0.5, L - 0.5)
plt.ylim(-L + 0.5, L - 0.5)

tps = 0


def update(iframe):
    global pp, ppp, tps
    # pp.set_data([XAmem[iframe][0],XBmem[iframe][0]],[XAmem[iframe][1],XBmem[iframe][1]])
    # plt.setp(pp,marker="o",linewidth=0)
    pp.set_data(XAmem[iframe][0], XAmem[iframe][1])
    plt.setp(pp, marker="o", markersize=20., color="b", linewidth=0)

    ppp.set_data(XBmem[iframe][0], XBmem[iframe][1])
    plt.setp(ppp, marker="o", markersize=20., color="r", linewidth=0)

    plt.title('t= ' + str(round(tps, 2)))
    if tps > Nt*dt:
        tps = 0
    else:
        tps += dt
    return pp, ppp


anim = animation.FuncAnimation(fig, update, frames=t.size, interval=0.1, repeat=True)
plt.show()

# Writer=animation.FFMpegWriter(fps=20)
# anim.save("test_cell.mp4",writer=Writer)

# Writer=animation.AVConvWriter(fps=20)
# anim.save("test_cell_bis.avi",writer=Writer)



