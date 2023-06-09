import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

A = np.array([[-0.0665, 8, 0, 0],
     [0, -3.663, 3.663, 0],
     [-6.86, 0, -13.736, -13.736],
     [0.6, 0, 0, 0]])
B = np.array([[0], [0], [13.7355], [0]])

C = np.eye(4)
D = np.zeros((4,1))
Ts = 0.01

Sys_d = cont2discrete((A,B,C,D), Ts, method = 'zoh')

Ad,Bd,Cd,Dd = Sys_d[0],Sys_d[1],Sys_d[2],Sys_d[3]

Q = np.eye(4)
R = np.eye(1)
#
# P = solve_discrete_are(Ad,Bd,Q,R)
# print(P)


T = 0.15
total_time = 60
num_p = round(total_time/T)
t_span = np.linspace(0,total_time,num_p)
x = np.zeros((num_p,4))
u = np.zeros((num_p,1))
x[0,:] = np.random.rand(4,)
W_C = np.random.rand(10,)
W_A = np.random.rand(4,)
phi_C = np.zeros((num_p,10))
beta = 0.001
alpha = 0.0001
gamma = 1

for k in range(num_p-1):
     # print(k)
     r = 0.5*np.matmul(np.matmul(x[k,:].T,Q),x[k,:])
     phi_C[k,:] = np.array(
          [x[k,0] ** 2, x[k,1] ** 2, x[k,2] ** 2, x[k,3] ** 2, x[k,0] * x[k,1], x[k,0] * x[k,2], x[k,0] * x[k,3],
           x[k,1] * x[k,2], x[k,1] * x[k,3], x[k,2] * x[k,3]]).reshape(10,)
     phi_A = np.array(x[k,:])
     del_phi_C = np.array([[2*x[k,0],0,0,0], [0,2*x[k,1],0,0], [0,0,2*x[k,2],0], [0,0,0,2*x[k,3]], [x[k,1],x[k,0],0,0],
           [x[k,2],0,x[k,0],0],[x[k,3],0,0,x[k,0]], [0,x[k,2],x[k,1],0],[0,x[k,3],0,x[k,1]],
          [0,0,x[k,3],x[k,2]]])
     # print(del_phi_C.shape)
     u[k] = np.matmul(W_A.T,phi_A)
     x[k+1,:] = np.matmul(Ad,x[k,:])+np.matmul(Bd,u[k])
     if k == 0:
          PHI = -phi_C[k,:]
     else:
          PHI = phi_C[k-1,:]-gamma*phi_C[k,:]
     W_C = W_C - alpha * PHI * (np.matmul(W_C.T,PHI)-r)
     W_A = W_A - beta * phi_A *(2*np.matmul(R*W_A.T,phi_A)+gamma * np.matmul(np.matmul(B.T,del_phi_C.T),W_C)).T
     print(x[k+1])
     print(W_A)
     print(np.matmul(W_C.T,phi_C[k,:]))