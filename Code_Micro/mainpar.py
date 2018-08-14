import numpy as np
import matplotlib.pyplot as plt
import initial_condition as ic
import brownian as bw
import potential as pt
import birthdeath as bd
import time
import matplotlib.animation as animation
from multiprocessing import Pool
import os

def test_process(index):
    
    if index==0:
        s=1.7
        nuAb=1e-5
        nuBb=1e-4
        logi=1
    if index==1:
        s=1.7
        nuAb=1e-5
        nuBb=1e-4
        logi=0
    if index==2:
        s=1.51
        nuAb=5e-5
        nuBb=1e-4
        logi=1
    if index==3:
        s=1.51
        nuAb=5e-5
        nuBb=1e-4
        logi=0
    if index==4:
        s=1.43
        nuAb=1e-4
        nuBb=1e-4
        logi=1
    if index==5:
        s=1.43
        nuAb=1e-4
        nuBb=1e-4
        logi=0
    if index==6:
        s=1.
        nuAb=1e-4
        nuBb=1e-4
        logi=1
    if index==7:
        s=1.
        nuAb=1e-4
        nuBb=1e-4
        logi=0
    if index==8:
        s=2.
        nuAb=1e-4
        nuBb=1e-4
        logi=1
    if index==9:
        s=2.
        nuAb=1e-4
        nuBb=1e-4
        logi=0
    if index==10:
        s=1.35
        nuAb=1e-4
        nuBb=5e-5
        logi=1
    if index==11:
        s=1.35
        nuAb=1e-4
        nuBb=5e-5
        logi=0
    if index==12:
        s=1.3
        nuAb=1e-4
        nuBb=1e-5
        logi=1
    if index==13:
        s=1.3
        nuAb=1e-4
        nuBb=1e-5
        logi=0
    
    # Parameters for the model

    KAA = 2.
    KAB = s
    KBA = 2*s
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
    
    NA = 250
    NB = 250
    mu = 1.
    
    b0A = nuAb*logi
    d0A = nuAb/10.*logi
    b0B = nuBb*logi
    d0B = nuBb/10.*logi
    
    thA = nuAb*0.75*logi
    thB = nuBb*0.75*logi
    
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
    
    np.save('results_case_'+str(index),[XAmem,XBmem])




if __name__=="__main__":
    cpus=os.cpu_count()
    p=Pool(cpus)
    p.map(test_process,[8,9])

