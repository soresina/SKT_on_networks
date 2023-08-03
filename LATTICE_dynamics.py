import sys
sys.path.append("/opt/homebrew/lib/python3.9/site-packages")

import networkx as nx
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
################################################################
# Network topology
# 2D-lattice
##m = 20
##n = 20
##G = nx.grid_2d_graph(m,n)

n=19
m=38
G = nx.triangular_lattice_graph(n,m)


spec = nx.laplacian_spectrum(G)
LG = nx.laplacian_matrix(G)
N=len(spec)
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
t=np.linspace(0,10000)
# initial condition
z0 = [us+np.multiply(np.random.rand(N)-0.5, 0.2*us), vs+np.multiply(np.random.rand(N)-0.5, 0.2*vs)]
z0= np.reshape(z0,2*N)
# solve ODE
z = odeint(model,z0,t) # rows<-- time, columns<-- variables (size=t*2N)
###############################################################
# plot results
###############################################################
# plot eigenvalues
plt.figure(1)
plt.plot([0,N],[lambda_1,lambda_1],'r:')
plt.plot([0,N],[lambda_2,lambda_2],'r:')
plt.plot([0,N],[lambda_thr,lambda_thr],'b:')
plt.plot(spec,'k*')
# plot dynamics
plt.figure(2)
for inodes in range(N):
    plt.plot(t,z[:,inodes],'b-')
    plt.plot(t,z[:,N+inodes],'r--')
plt.plot([0,t[-1]],[us,us],'b:')
plt.plot([0,t[-1]],[vs,vs],'r:')
plt.ylabel('u,v')
plt.xlabel('time')
# plot steady state
plt.figure(3)
plt.plot(z[-1,0:N],'b.')
plt.plot([0,N-1],[us,us],'b:')
plt.plot(z[-1,N:2*N-1],'r.')
plt.plot([0,N-1],[vs,vs],'r:')
# the network
plt.figure(4)
colors = [  
            '#666666', #gray
            '#1b9e77', #green
            '#e7298a'  #magenta
            ]
pos = dict( (n, n) for n in G.nodes() )
##nx.set_node_attributes(G, 2 , name='U')#G.nodes[1,1]['U']
##nx.set_node_attributes(G, z[-1,N:2*N-1], name='V')
n_colors = z[-1,0:N]#[G.nodes[node]['U'] for node in G.nodes()] #plt.cm.Blues

nx.draw_networkx(G, pos=pos, node_size=10, node_color=n_colors, cmap='viridis', with_labels=False, edge_color='#666666', width=0.5)
sc =nx.draw_networkx_nodes(G, pos=pos, node_size=10, node_color=n_colors, cmap='viridis', with_labels=False, edge_color='#666666', width=0.5)
cbar = plt.colorbar(sc)
plt.axis('equal')
plt.axis('off')

plt.figure(5)
n_colors_v = z[-1,N:2*N]
nx.draw_networkx(G, pos=pos, node_size=10, node_color=n_colors_v, cmap='plasma', with_labels=False, edge_color='#666666', width=0.5)
scv =nx.draw_networkx_nodes(G, pos=pos, node_size=10, node_color=n_colors_v, cmap='viridis', with_labels=False, edge_color='#666666', width=0.5)
cbarv = plt.colorbar(scv)
plt.axis('equal')
plt.axis('off')

### 3D steady state 
##fig=plt.figure(6)
##ax = fig.add_subplot(111, projection='3d')
##ss=0
##for ii in range(n):
##    for jj in range(m):
##        U=[ii,jj,z[-1,ss]]
##        ax.scatter(U[0],U[1],U[2])        
##        ss=ss+1
##fig=plt.figure(7)
##ax = fig.add_subplot(111, projection='3d')
##ss=0
##for ii in range(n):
##    for jj in range(m):
##        V=[ii,jj,z[-1,N+ss]]
##        ax.scatter(V[0],V[1],V[2])        
##        ss=ss+1

plt.show()
