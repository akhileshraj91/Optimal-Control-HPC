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
total_time = 10
num_p = round(total_time/T)
t_span = np.linspace(0,total_time,num_p)
x = np.random.rand(num_p,4)
u = np.zeros((num_p,1))
x[0,:] = np.random.rand(4,)
# W_C = np.random.rand(10,)
W_C = np.zeros((10,))
# W_A = np.random.rand(4,)
phi_C = np.zeros((num_p,10))
# beta = 0.001
# alpha = 0.0001
gamma = 1

BATCH_SIZE = 20
X = np.zeros((BATCH_SIZE,phi_C.shape[1]))
Y = np.zeros((BATCH_SIZE,1))
batch = 0
K = np.array([1,1,1,1])

for k in range(num_p-1):

    phi_C[k,:] = np.array(
        [x[k,0] ** 2, x[k,1] ** 2, x[k,2] ** 2, x[k,3] ** 2, x[k,0] * x[k,1], x[k,0] * x[k,2], x[k,0] * x[k,3],
        x[k,1] * x[k,2], x[k,1] * x[k,3], x[k,2] * x[k,3]]).reshape(10,)

    del_phi_C = np.array([[2*x[k,0],0,0,0], [0,2*x[k,1],0,0], [0,0,2*x[k,2],0], [0,0,0,2*x[k,3]], [x[k,1],x[k,0],0,0],
        [x[k,2],0,x[k,0],0],[x[k,3],0,0,x[k,0]], [0,x[k,2],x[k,1],0],[0,x[k,3],0,x[k,1]],
        [0,0,x[k,3],x[k,2]]])

    if k == 0:
        PHI = -gamma*phi_C[k,:]
        # u[k] = -np.matmul(K,x[k,:])
    else:
        PHI = phi_C[k-1,:]-gamma*phi_C[k,:]
    u[k] = -gamma/2*(np.matmul(np.matmul(np.matmul(np.linalg.inv(R),B.T),del_phi_C.T),W_C))
    print(u[k],x[k,:])
    r = 0.5*np.matmul(np.matmul(x[k,:].T,Q),x[k,:])+u[k]*R*u[k]

    x[k+1,:] = np.matmul(Ad,x[k,:])+np.matmul(Bd,u[k])


    X[batch,:] = PHI 
    Y[batch,:] = r[0][0]
    if batch >= BATCH_SIZE-1:
        # print(X.shape,"\n",W_C.shape,"\n",Y.shape,"\n",r,"\n",batch,"____________")
        W_C = np.dot((np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)),Y)
        X = np.zeros((BATCH_SIZE,phi_C.shape[1]))
        Y = np.zeros((BATCH_SIZE,1))
        # print(np.linalg.norm(X),np.linalg.norm(Y))
        batch = 0
        # print(W_C)
    else:
        batch = batch+1


        #  W_C = W_C - alpha * PHI * (np.matmul(W_C.T,PHI)-r)
        #  W_A = W_A - beta * phi_A *(2*np.matmul(R*W_A.T,phi_A)+gamma * np.matmul(np.matmul(B.T,del_phi_C.T),W_C)).T
    # print(x[k+1])
    # print(W_C)
    # print(np.matmul(W_C.T,phi_C[k,:]))