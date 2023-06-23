import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve_discrete_are


a_J = 10**(-6)
a_u = 0.1



A = np.array([[0,-0.8],[0.8,1.8]])
B = np.array([[0],[-1]])
Q = np.eye(2)
R = np.array([[1]])
n = A.shape[1]
m = B.shape[1]
P = solve_discrete_are(A,B,Q,R)
K = np.matmul(np.matmul(np.matmul(np.linalg.inv(R+np.matmul(np.matmul(B.T,P),B)),B.T),P),A)
print(P,"____",K)

T = 0.01
tspan = 1
num_t = int(tspan/T)
T_INS = np.linspace(0,tspan,num_t)
# x = np.zeros((num_t,n))
# x[0,:] = np.random.rand(n,)
# u = np.zeros((num_t,m))
# for k,t in enumerate(T_INS[:-1]):
#     u[k,:] = -np.matmul(K,x[k,:]) 
#     x[k+1,:] = np.matmul(A,x[k,:])+np.matmul(B,u[k,:])

# fig,axs = plt.subplots(2,1)

# axs[0].plot(T_INS,x)
# axs[0].set_xlabel("time[s]")
# axs[0].set_ylabel("states")
# axs[0].grid(True)
# axs[1].plot(T_INS,u)
# axs[1].set_xlabel("time[s]")
# axs[1].set_ylabel("control law")
# axs[1].grid(True)
# plt.show()



x = np.random.rand(num_t,n)
u = np.random.rand(num_t,m)
admissible_gain = np.array([[0.5,1.4]])
PHI = np.zeros((15,))
THETA = np.zeros((12,))
THETA[0] = 0.5
THETA[1] = 1.4
sigma = np.zeros((num_t,15))
neu = np.zeros((num_t,12))
delta_sigma = np.zeros_like(sigma)
buffer_size = 15
r = np.zeros((num_t,1))
for k,t in enumerate(T_INS[:-1]):
    sigma[k,:] = np.array([x[k,0]**2,x[k,0]*x[k,1],x[k,1]**2,x[k,0]**4,x[k,0]**3*x[k,1],x[k,0]**2*x[k,1]**2,
                   x[k,0]*x[k,1]**3,x[k,1]**4,x[k,0]**6,x[k,0]**5*x[k,1],x[k,0]**4*x[k,1]**2,x[k,0]**3*x[k,1]**3,
                   x[k,0]**2*x[k,1]**4,x[k,0]*x[k,1]**5,x[k,1]**6])
    neu[k,:] = np.array([x[k,0],x[k,1],x[k,0]**3,x[k,0]**2*x[k,1],x[k,0]*x[k,1]**2,x[k,1]**3,x[k,0]**5,
                      x[k,0]**4*x[k,1],x[k,0]**3*x[k,1]**2,x[k,0]**2*x[k,1]**3,x[k,0]*x[k,1]**4,x[k,1]**5])
    r[k,:] = np.matmul(np.matmul(x[k,:].T,Q),x[k,:]) + np.matmul(np.matmul(u[k,:].T,R),u[k,:])  
    if k != 0:
        delta_sigma[k,:] = sigma[k,:] - sigma[k-1,:]
    else:
        delta_sigma[k,:] = sigma[k,:]
    
    if k>=buffer_size:
        X_K = delta_sigma[k-buffer_size:k,:]
        Y_K = r[k-buffer_size:k,:]
        E_JK = Y_K + np.matmul(PHI.T,X_K)
        PHI = np.matmul(np.matmul(X_K,np.linalg.inv(np.matmul(X_K.T,X_K))),(a_J*E_JK.T-Y_K.T))
    J = np.matmul(PHI.T,sigma[k,:])
    U = np.matmul(THETA.T,neu[k,:])