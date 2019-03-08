import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

index = 0
nbr_of_itr = 50

dataA = np.load("data/results_A_case_"+str(index)+"_"+str(nbr_of_itr)+".npz")
dataB = np.load("data/results_B_case_"+str(index)+"_"+str(nbr_of_itr)+".npz")
nuAb = 0.
nuBb = 0.
nuAd = 0.
nuBd = 0.
if index == 0:
    s = 2.
    nuAb = 0.011
    nuBb = 0.022
    nuAd = 0.001
    nuBd = 0.002
    logi = 1
if index == 1:
    s = 2.
    nuAb = 0.011
    nuBb = 0.022
    nuAd = 0.001
    nuBd = 0.002
    logi = 0
if index == 2:
    s = 3.
    nuAb = 0.011
    nuBb = 0.022
    nuAd = 0.001
    nuBd = 0.002
    logi = 1
if index == 3:
    s = 3.
    nuAb = 0.011
    nuBb = 0.022
    nuAd = 0.001
    nuBd = 0.002
    logi = 0
if index == 4:
    s = 4.
    nuAb = 0.011
    nuBb = 0.022
    nuAd = 0.001
    nuBd = 0.002
    logi = 1
if index == 5:
    s = 4.
    nuAb = 0.011
    nuBb = 0.022
    nuAd = 0.001
    nuBd = 0.002
    logi = 0

N=len(dataA.files)
dt=0.1
T=nbr_of_itr*dt
L=7.5
XAmem=[]
XBmem=[]

# for i in range(0,N,10):
#     print(i)
#     XAmem.append(dataA["arr_"+str(i)])
#     XBmem.append(dataB["arr_"+str(i)])
    

XAmem.append(dataA["arr_"+str(0)])
XBmem.append(dataB["arr_"+str(0)])

XAmem.append(dataA["arr_"+str(N-1)])
XBmem.append(dataB["arr_"+str(N-1)])
    
NA = XAmem[0].shape[0]
NB = XBmem[0].shape[0]
MaxN = np.max([NA, NB])
MinN = np.min([NA, NB])


# Plot à l'instant final avec affichage aléatoire pour l'overlapping
    
plt.figure(0)
randomlistA = np.random.permutation(range(NA))
randomlistB = np.random.permutation(range(NB))

for i in range(MaxN):
    if i<MinN:
            pp=plt.plot(XAmem[0][randomlistA[i]], XAmem[1][randomlistA[i]])
            plt.setp(pp, marker="o", markersize=20., color="b", linewidth=0,alpha=0.8)
            ppp = plt.plot(XBmem[0][randomlistB[i]], XBmem[1][randomlistB[i]])
            plt.setp(ppp, marker="o", markersize=20., color="r", linewidth=0,alpha=0.5)
    if i>=MinN:
        if MaxN == NA:
            pp = plt.plot(XAmem[0][randomlistA[i]], XAmem[1][randomlistA[i]])
            plt.setp(pp, marker="o", markersize=20., color="b", linewidth=0,alpha=0.8)
        elif MaxN == NB:
            ppp = plt.plot(XBmem[0][randomlistB[i]], XBmem[1][randomlistB[i]])
            plt.setp(ppp, marker="o", markersize=20., color="r", linewidth=0,alpha=0.5)
plt.title('t= ' + str(T)+", NA="+str(XAmem[0].shape[0])+", NB="+str(XBmem[0].shape[0])+", Case "+str(index))
# plt.savefig("Tests_results/case_"+str(index)+"_"+str(nbr_of_itr)+"_final.png")
plt.show()

# Plot à l'instant final

# plt.figure(1)
# pp=plt.plot(XAmem[-1][0], XAmem[-1][1])
# plt.setp(pp, marker="o", markersize=20., color="b", linewidth=0, alpha=0.8)
#
# ppp=plt.plot(XBmem[-1][0], XBmem[-1][1])
# plt.setp(ppp, marker="o", markersize=20., color="r", linewidth=0, alpha=0.3)
#
# plt.title('t= ' + str(T)+", NA="+str(XAmem[-1].shape[1])+", NB="+str(XBmem[-1].shape[1])+", Case "+str(index))
# plt.savefig("Tests_results/case_"+str(index)+"_final.png")
# plt.show()

# ", $s=$ "+str(s)+ ", $b_0^{A}=$ "+str(nuAb)+ ", $b_0^{B}=$ "+str(nuBb)+ ", $d_0^{A}=$ "+str(nuAd)+ ", $d_0^{B}=$ "+str(nuBd)+ ", logistic="+str(logi))
# Plot à l'instant final séparés

# plt.figure(2)
# plt.subplot(121)
# pp=plt.plot(XAmem[-1][0], XAmem[-1][1])
# plt.setp(pp, marker="o", markersize=20., color="b", linewidth=0, alpha=0.8)
#
# plt.subplot(122)
# ppp=plt.plot(XBmem[-1][0], XBmem[-1][1])
# plt.setp(ppp, marker="o", markersize=20., color="r", linewidth=0, alpha=0.78)
#
# plt.title('t= ' + str(T)+", NA="+str(XAmem[-1].shape[1])+", NB="+str(XBmem[-1].shape[1])+", Case "+str(index))
# plt.show()

# # Animation
# 
# fig = plt.figure(3)
# t = np.arange(0., T, dt)
# # axis = fig.add_subplot(111, xlim=(-L, L), ylim=(-L, L))
# pp, ppp = plt.plot([], [], [], [])
# plt.xlim(-L + 0.5, L - 0.5)
# plt.ylim(-L + 0.5, L - 0.5)
# 
# tps = 0
# 
# 
# def update(iframe):
#     global pp, ppp, tps
#     # pp.set_data([XAmem[iframe][0],XBmem[iframe][0]],[XAmem[iframe][1],XBmem[iframe][1]])
#     # plt.setp(pp,marker="o",linewidth=0)
#     pp.set_data(XAmem[iframe][0], XAmem[iframe][1])
#     plt.setp(pp, marker="o", markersize=20., color="b", linewidth=0)
# 
#     ppp.set_data(XBmem[iframe][0], XBmem[iframe][1])
#     plt.setp(ppp, marker="o", markersize=20., color="r", linewidth=0)
# 
#     plt.title('t= ' + str(round(tps, 2)))
#     if tps > N*dt:
#         tps = 0
#     else:
#         tps += dt
#     return pp, ppp
# 
# 
# anim = animation.FuncAnimation(fig, update, frames=t.size, interval=0.1, repeat=True)
# plt.show()
    
