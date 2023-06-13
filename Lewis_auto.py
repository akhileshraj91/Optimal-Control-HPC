import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt

A = np.array([[0.8,1],[1.1,2]])
B = np.array([[0.2], [1.4]])
W1 = np.array([[0.7,0],[-1,-0.5]])
W2 = np.array([[-1],[0.8]])
W = np.eye(2)

Q = np.eye(2)
R = np.eye(1)
gamma = 0.7
#
# P = solve_discrete_are(A,B,Q,R)
# print(P)


T = 0.01
total_time = 20
num_p = round(total_time/T)
t_span = np.linspace(0,total_time,num_p)

L_0 = np.array([-1.4,-2.1])
X_0 = np.random.rand(2,2)
N = 5
i_max = 20
num_mean = 5
epsilon = 10**(-2)

L = np.zeros((N, 1, 2))
L[0, :, :] = L_0
x = np.zeros((N,2))
d = np.zeros((N,))
w = np.zeros((N,))
for i in range(0,i_max):
     x_0 = np.random.multivariate_normal([0,0],X_0)
     PHI = 0
     PSI = 0
     GAMMA = 0
     for q in range(1,num_mean):
          u = np.zeros((N,))
          x[0,:] = np.random.rand(2,)
          for k in range(1,N):
               d[k] = np.random.rand(1,)
               w[k] = np.random.rand(1,)
               u[k] = np.matmul(L[i,:,:],x[k])
               x[k+1,:] = np.matmul(A,x[k,:].reshape(2,1)) + B*u[k] + (np.matmul(W1,x[k,:].reshape(2,1)) + B*u[k])*d[k]+w[k]
               PHI_i =



#
#
# x = np.zeros((num_p,4))
# u = np.zeros((num_p,1))
# x[0,:] = 2*np.random.rand(4,)-1
# W_C = np.random.rand(num_p,10)
# W_A = np.random.rand(4,)
# phi_C = np.zeros((num_p,10))
# beta = 0.01
# alpha = 0.001
# gamma = 1
# u_real = np.zeros((num_p,1))
# for k in range(num_p-1):
#      # print(k)
#      r = 0.5*np.matmul(np.matmul(x[k,:].T,Q),x[k,:])
#      phi_C[k,:] = np.array(
#           [x[k,0] ** 2, x[k,1] ** 2, x[k,2] ** 2, x[k,3] ** 2, x[k,0] * x[k,1], x[k,0] * x[k,2], x[k,0] * x[k,3],
#            x[k,1] * x[k,2], x[k,1] * x[k,3], x[k,2] * x[k,3]]).reshape(10,)
#      # phi_A = np.array(x[k,:])
#      del_phi_C = np.array([[2*x[k,0],0,0,0], [0,2*x[k,1],0,0], [0,0,2*x[k,2],0], [0,0,0,2*x[k,3]], [x[k,1],x[k,0],0,0],
#            [x[k,2],0,x[k,0],0],[x[k,3],0,0,x[k,0]], [0,x[k,2],x[k,1],0],[0,x[k,3],0,x[k,1]],
#           [0,0,x[k,3],x[k,2]]])
#      # print(del_phi_C.shape)
#      # u[k] = np.matmul(W_A.T,phi_A)
#      u[k] = -gamma/2*(np.linalg.inv(R)*np.matmul(np.matmul(B.T,del_phi_C.T),W_C[k,:]))
#      u_real[k] = -gamma / 2 * (np.linalg.inv(R) * np.matmul(np.matmul(B.T, P), x[k,:]))
#
#      x[k+1,:] = np.matmul(Ad,x[k,:])+np.matmul(Bd,u[k])
#      if k == 0:
#           PHI = -phi_C[k,:]
#      else:
#           PHI = phi_C[k-1,:]-gamma*phi_C[k,:]
#      W_C[k+1,:] = W_C[k,:] - alpha * PHI * (np.matmul(W_C[k,:].T,PHI)-r)
#      # W_A = W_A - beta * phi_A *(2*np.matmul(R*W_A.T,phi_A)+gamma * np.matmul(np.matmul(B.T,del_phi_C.T),W_C)).T
#      # print(x[k+1])
#      # print(W_A)
#      # print(np.matmul(W_C[k,:].T,phi_C[k,:]))
# print(np.matmul(del_phi_C.T,W_C[-1,:]))
# fig,axs = plt.subplots(3,1)
# axs[0].plot(t_span,x)
# axs[1].plot(t_span,u)
# axs[1].plot(t_span,u_real)
# axs[2].plot(t_span,W_C)
# plt.show()