import numpy as np
from scipy.integrate import odeint
from sklearn.datasets import make_spd_matrix
import random
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

# Define the ODE system
a_1 = 0.2
a_2 = 0.1
Q = 5*np.eye(3)
R = np.array([[1]])
# T = 0.01
# F2 = 0.5 * np.eye(6)
F2_dash = 2 * np.random.rand(6, 6)
spd = make_spd_matrix(n_dim=6, random_state=1)
F2 = 1 * np.dot(spd, spd.T)
# print(np.linalg.eig(F2))
F1 = np.random.rand(6,)
# F1 = 0.1 * np.ones((6,))
q = 1
def x_dot(X, t, a, b, u):
    # print(X.shape)
    x = X[0:3,]
    x_1 = x[0,]
    x_2 = x[1,]
    x_3 = x[2,]
    W_1_hat = X[3:9,]
    W_2_hat = X[9:15,]
    del_phi_1 = np.array([[2*x_1,0,0],[2*x_2,2*x_1,0],[2*x_3,0,2*x_1],[0,2*x_2,0],[0,2*x_3,2*x_2],[0,0,2*x_3]])
    n_t = 2*np.exp(-0.009*t)*(np.sin(t)**2*np.cos(t)+np.sin(2*t)**2*np.cos(0.1*t)+np.sin(-1.2*t)**2*np.cos(0.5*t)+np.sin(t)**5)
    u = -0.5*np.matmul(np.matmul(np.matmul(np.linalg.inv(R), B.T), del_phi_1.T),W_2_hat)+n_t
    cost = np.matmul(np.matmul(x.T,Q),x)+np.matmul(np.matmul(u.T,R),u)
    sigma_2 = np.matmul(del_phi_1,np.matmul(a,x)+np.matmul(b,u))
    sigma_2_bar = sigma_2/(np.matmul(sigma_2.T,sigma_2)+1)
    M = sigma_2/(np.matmul(sigma_2.T,sigma_2)+1)**2
    phi_1 = np.array([x_1**2,2*x_1*x_2,2*x_1*x_3,x_2**2,2*x_2*x_3,x_3**2])
    W_1_hat_dot = -a_1 * (M) * (np.matmul(sigma_2.T,W_1_hat)+cost)
    D_1 = np.matmul(np.matmul(np.matmul(np.matmul(del_phi_1,b),np.linalg.inv(R)),b.T),del_phi_1.T)


    ms = np.matmul(sigma_2.T,sigma_2)+1
    # check_mat = np.array([[q*np.eye(3), np.zeros((1,1)), np.zeros((6,6))],[np.zeros((3,3)), np.eye(1), -0.5*F1-(1/(8*ms))*np.matmul(D_1,W1_hat).T],[np.zeros(6,6),-0.5*F1-(1/(8*ms))*np.matmul(D_1,W1_hat),F2-(1/8)*(np.matmul(np.matmul(D_1,W_1_hat),M.T)+np.matmul(M,W_1_hat.T)*D_1)]])

    # W_2_hat_dot = -a_2* ((F2*W_2_hat)-(F1*np.matmul((sigma_2/(np.matmul(sigma_2.T,sigma_2)+1)).T,W_1_hat))-0.25*np.matmul(D_1,W_2_hat*(np.matmul(M.T,W_1_hat))))
    W_2_hat_dot = -a_2*(np.matmul(F2,W_2_hat)-F1*np.matmul(sigma_2_bar.T,W_1_hat)-0.25*np.matmul(D_1,W_2_hat*(np.matmul(M.T,W_1_hat))))
    V_hat = np.matmul(W_1_hat.T, phi_1)
    # print(V_hat)
    # print(V_hat.shape,sigma_1.shape,W_1_hat_dot.shape,cost)
    # print(W_1_hat)
    x_dot = np.dot(A, x) + np.dot(B, u)
    return_vec = np.concatenate((x_dot,W_1_hat_dot,W_2_hat_dot))
    return return_vec

# Define the initial conditions and parameters
x0 = np.array([1,1,1]) # initial state of x
A = np.array([[-1.01887, 0.90506, -0.00215],[0.82225, -1.07741, -0.17555],[0,0,-1]]) # parameter a
B = np.array([[0],[0],[1]]) # parameter b
u = np.array([1.0]) # input u
W1_hat = np.random.rand(6,1).ravel()
W2_hat = np.random.rand(6,1).ravel()
# print(x0.shape,W1_hat.shape)
initial_value = np.concatenate((x0,W1_hat,W2_hat))
# Define the time vector for integration
t = np.linspace(0, 800, 1000)

# Solve the ODE system using odeint
x = odeint(x_dot, initial_value, t, args=(A, B, u))
W1_data = x[:,3:8]
W2_data = x[:,9:14]
print(W1_data[0,:],'\n',W1_data[-1,:])
print(W2_data[0,:],'\n',W2_data[-1,:])
# Plot the solution
fig,axes = plt.subplots(3,1)
axes[0].plot(t, x[:,1:3])
axes[0].set_xlabel('Time')
axes[0].set_ylabel('x')
axes[1].plot(t,W1_data)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('W1')
axes[2].plot(t,W2_data)
axes[2].set_xlabel('Time')
axes[2].set_ylabel('W2')
plt.show()


# P = solve_continuous_are(A, B, Q, R)
# print(P)
# print(B)
# print(np.matmul(np.matmul(np.linalg.inv(R),B.T),P))



