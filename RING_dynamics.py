import sys
sys.path.append("/opt/homebrew/lib/python3.9/site-packages")

from smallworld.draw import draw_network
from smallworld import get_smallworld_graph
import networkx as nx
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 
################################################################
# Network topology
# 2K-regular ring
N = 100
k_over_2 = 10
p = 0
G = get_smallworld_graph(N, k_over_2, p)

spec = nx.laplacian_spectrum(G)
LG = nx.laplacian_matrix(G)
################################################################
# Parameter set
# weak competition
r1 = 5
r2 = 2
a1 = 3
a2 = 3
b1 = 1
b2 = 1
d12 = 3
d21 = 0
d = 0.03
# equilibrium
vs = (r2*a1-r1*b2)/(a1*a2-b1*b2) 
us = (r1*a2-r2*b1)/(a1*a2-b1*b2)
# thresholds
# Jacobian
traceJs = -a1*us-a2*vs
detJs = us*vs*(a1*a2-b1*b2)
# alpha and beta
alpha = (2*b2*us-r2)*vs;
beta = (2*b1*vs-r1)*us;
# threshold on lambda
lambda_thr = detJs/(d12*alpha+d21*beta)
# range lambda
A = d*(d+d12*vs+d21*us)
B = d12*alpha+d21*beta+d*traceJs
C = detJs
lambda_1 = (B+np.sqrt(B**2-4*A*C))/(2*A)
lambda_2 = (B-np.sqrt(B**2-4*A*C))/(2*A)
################################################################
# Integration
# function that returns dz/dt
def model(z,t):
    u=z[0:N]
    v=z[N:2*N]
    uip = u*(r1-np.multiply(a1,u)-np.multiply(b1,v))-d*LG*u-d12*LG*np.multiply(u,v)
    vip = v*(r2-np.multiply(b2,u)-np.multiply(a2,v))-d*LG*v-d21*LG*np.multiply(u,v)
    F = np.reshape([uip,vip],2*N)
    return F
# time
t=np.linspace(0,300,6000)
# initial condition
z0 = [us+np.multiply(np.random.rand(N)-0.5, 0.2*us), vs+np.multiply(np.random.rand(N)-0.5, 0.2*vs)]
z0= np.reshape(z0,2*N)
# solve ODE
z = odeint(model,z0,t)
# abundances on the network
tot_u=np.sum(z[-1,0:N])
tot_us=np.multiply(us,N)
tot_v=np.sum(z[-1,N:2*N-1])
tot_vs=np.multiply(vs,N)
tot_u_p=np.divide(tot_u-tot_us,tot_us)
tot_v_p=np.divide(tot_v-tot_vs,tot_vs)
print(tot_u)
print(tot_us)
print(tot_v)
print(tot_vs)
print(tot_u_p)
print(tot_v_p)
###############################################################
# plot results
###############################################################
str_d=str(d)
K=k_over_2
##name1='RING_d'+str_d[2:len(str_d)]+'_eig.eps'
name2='RING_N'+str(N)+'_K'+str(K)+'_dyn.eps'
name3='RING_N'+str(N)+'_K'+str(K)+'_steady.eps'

col_u_dark='darkmagenta'
col_u_light='pink'
col_v_dark='darkorange'
col_v_light='moccasin'
### plot eigenvalues
##plt.figure(1)
##plt.plot([0,N],[lambda_1,lambda_1],'r:')
##plt.plot([0,N],[lambda_2,lambda_2],'r:')
##plt.plot([0,N],[lambda_thr,lambda_thr],'b:')
##plt.plot(spec,'k*')
##plt.savefig(name1)
# plot dynamics
plt.figure(2)
fig, axs = plt.subplots(2)
axs[0].plot([0,t[-1]],[us,us],':',color=col_u_dark)
for inodes in range(N):
    axs[0].plot(t,z[:,inodes],'-',color=col_u_light)    
axs[0].plot(t,z[:,0],'-', color=col_u_dark)    

axs[1].plot([0,t[-1]],[vs,vs],':',color=col_v_dark)
for inodes in range(N):
    axs[1].plot(t,z[:,N+inodes],'-',color=col_v_light)
axs[1].plot(t,z[:,N],'-',color=col_v_dark)    
plt.savefig(name2)

#plt.ylabel('u,v')
#plt.xlabel('time')
# plot steady state

plt.figure(3)
fig, axs = plt.subplots(2)
axs[0].plot(z[-1,0:N],'.',color=col_u_dark)
axs[0].plot([0,N-1],[us,us],':',color=col_u_dark)
axs[1].plot(z[-1,N:2*N-1],'.',color=col_v_dark)
axs[1].plot([0,N-1],[vs,vs],':',color=col_v_dark)
fig.savefig(name3)
plt.show()
