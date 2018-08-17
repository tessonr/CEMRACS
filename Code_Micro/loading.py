import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

index=13

dataA=np.load("results_A_case_"+str(index)+".npz")
dataB=np.load("results_B_case_"+str(index)+".npz")

N=len(dataA.files)
dt=0.1
T=0.1*(N-1)
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
    
    
# Plot à l'instant initial
    
plt.figure(0)
pp=plt.plot(XAmem[0][0], XAmem[0][1])
plt.setp(pp, marker="o", markersize=20., color="b", linewidth=0)

ppp=plt.plot(XBmem[0][0], XBmem[0][1])
plt.setp(ppp, marker="o", markersize=20., color="r", linewidth=0)

plt.title('t= ' + str(0)+", NA="+str(XAmem[0].shape[1])+", NB="+str(XBmem[0].shape[1])+", Case "+str(index))
plt.show()

# Plot à l'instant final

plt.figure(1)
pp=plt.plot(XAmem[-1][0], XAmem[-1][1])
plt.setp(pp, marker="o", markersize=20., color="b", linewidth=0)

ppp=plt.plot(XBmem[-1][0], XBmem[-1][1])
plt.setp(ppp, marker="o", markersize=20., color="r", linewidth=0)

plt.title('t= ' + str(T)+", NA="+str(XAmem[-1].shape[1])+", NB="+str(XBmem[-1].shape[1])+", Case "+str(index))
plt.show()

# Plot à l'instant final séparés

plt.figure(2)
plt.subplot(121)
pp=plt.plot(XAmem[-1][0], XAmem[-1][1])
plt.setp(pp, marker="o", markersize=20., color="b", linewidth=0)

plt.subplot(122)
ppp=plt.plot(XBmem[-1][0], XBmem[-1][1])
plt.setp(ppp, marker="o", markersize=20., color="r", linewidth=0)

plt.title('t= ' + str(T)+", NA="+str(XAmem[-1].shape[1])+", NB="+str(XBmem[-1].shape[1])+", Case "+str(index))
plt.show()

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
    
