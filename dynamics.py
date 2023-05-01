import numpy as np
from scipy.integrate import odeint
import random
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

# Define the ODE system
a_1 = 0.1
Q = np.eye(3)
R = np.array([[1]])
T = 0.01
def x_dot(X, t, a, b, u):
    # print(X.shape)
    x = X[0:3,]
    x_1 = x[0,]
    x_2 = x[1,]
    x_3 = x[2,]
    W_1_hat = X[3:,]
    cost = np.matmul(np.matmul(x.T,Q),x)+np.matmul(np.matmul(u.T,R/T),u)
    del_phi_1 = np.array([[2*x_1,0,0],[2*x_1,2*x_2,0],[2*x_3,0,2*x_1],[0,2*x_2,0],[0,2*x_3,2*x_2],[0,0,2*x_3]])
    sigma_1 = np.matmul(del_phi_1,np.matmul(a,x)+np.matmul(b,u))
    phi_1 = np.array([x_1**2,2*x_1*x_2,2*x_1*x_3,x_2**2,2*x_2*x_3,x_3**2])
    W_1_hat_dot = -a_1 * sigma_1/(np.matmul(sigma_1.T,sigma_1)+1)**2*(np.matmul(sigma_1.T,W_1_hat)+cost)
    u = -0.5*T*np.matmul(np.matmul(np.matmul(np.linalg.inv(R), B.T), del_phi_1.T),W_1_hat)
    V_hat = np.matmul(W_1_hat.T, phi_1)
    # print(V_hat.shape,sigma_1.shape,W_1_hat_dot.shape,cost)
    # print(W_1_hat)
    n_t = 2*np.exp(-0.009*t)*(np.sin(t)**2*np.cos(t)+np.sin(2*t)**2*np.cos(0.1*t)+np.sin(-1.2*t)**2*np.cos(0.5*t)+np.sin(t)**5)
    x_dot = np.dot(A, x) + np.dot(B, (u+n_t))
    return_vec = np.concatenate((x_dot,W_1_hat_dot))
    return return_vec

# Define the initial conditions and parameters
x0 = np.array([1,1,1]) # initial state of x
A = np.array([[-1.01887, 0.90506, -0.00215],[0.82225, -1.07741, -0.17555],[0,0,-1]]) # parameter a
B = np.array([[0],[0],[1]]) # parameter b
u = np.array([1.0]) # input u
W1_hat = np.zeros((6,1)).ravel()
# print(x0.shape,W1_hat.shape)
initial_value = np.concatenate((x0,W1_hat))
# Define the time vector for integration
t = np.linspace(0, 800, 1000)

# Solve the ODE system using odeint
x = odeint(x_dot, initial_value, t, args=(A, B, u))
W1_data = x[:,3:]
print(W1_data[0,:],'\n',W1_data[-1,:])
# Plot the solution
fig,axes = plt.subplots(2,1)
axes[0].plot(t, x[:,1:3])
axes[0].set_xlabel('Time')
axes[0].set_ylabel('x')
axes[1].plot(t,W1_data)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('W1')
plt.show()


# P = solve_continuous_are(A, B, Q, R)
# print(P)
# print(B)
# print(np.matmul(np.matmul(np.linalg.inv(R),B.T),P))



