import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def cot(x):
    return 1.0/np.tan(x)

beta = np.linspace(np.pi/100,0.49*np.pi,100)
rho = np.pi/6
phi = np.pi/6
delta = np.pi/6
gamma = 100.0
c = 1.0
g = 9.8
d = 1.0
q = 1.0 


N_g= np.empty(beta.shape)
N_c = np.empty(beta.shape)
N_q = np.empty(beta.shape)

for i in range(len(beta)):
    N_g[i] = (cot(rho) + cot(beta[i]))/(2*(np.cos(rho+delta) + np.sin(rho+delta)*cot(beta[i]+phi)))
    N_c[i]  = (1+cot(beta[i])*cot(beta[i]+phi))/(np.cos(rho+delta) + np.sin(rho+delta)*cot(beta[i]+phi))
    N_q[i]  = (cot(rho) + cot(beta[i]))/(np.cos(rho+delta) + np.sin(rho+delta)*cot(beta[i]+phi))

fig = plt.figure()
ax = fig.gca()
ax.plot(beta,N_g)
ax.plot(beta,N_c)
ax.plot(beta,N_q)
ax.set_xlabel('time (dimensionless)')
ax.set_ylabel('force (normalized)')
ax.legend(['N_g','N_c','N_q'])
plt.show()
