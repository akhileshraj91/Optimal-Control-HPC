import numpy as np
from scipy.integrate import odeint
from sklearn.datasets import make_spd_matrix
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

#constants
a_1 = 0.01
a_2 = 0.001
Q = np.eye(2)
R = np.array([[1]])
F2_dash = 1 * np.random.rand(3, 3)
spd = make_spd_matrix(n_dim=3, random_state=1)
F2 = 2 * np.dot(spd, spd.T)
F1 = 2 * np.random.rand(3,)
# F1 = 0.1 * np.ones((6,))
# F2 = 0.5 * np.eye(6)
q = 1

# Define the ODE system
def x_dot(X, t):
    x = X[0:2,]
    x_1 = x[0,]
    x_2 = x[1,]
    f = np.array([[-x_1+x_2],[-0.5*x_1-0.5*x_2*(1-(np.cos(2*x_1)+2)**2)]]).ravel()
    g = np.array([[0],[np.cos(2*x_1)+2]])
    W_1_hat = X[2:5,]
    W_2_hat = X[5:8,]
    del_phi_1 = np.array([[2*x_1,0],[x_2,x_1],[0,2*x_2]])
    # n_t = 2*np.exp(-0.009*t)*(np.sin(t)**2*np.cos(t)+np.sin(2*t)**2*np.cos(0.1*t)+np.sin(-1.2*t)**2*np.cos(0.5*t)+np.sin(t)**5)
    n_t = 0.01*np.random.normal(0,0.01)
    u = -0.5*np.matmul(np.matmul(np.matmul(np.linalg.inv(R), g.T), del_phi_1.T),W_2_hat)+n_t
    cost = np.matmul(np.matmul(x.T,Q),x)+np.matmul(np.matmul(u.T,R),u)
    sigma_2 = np.matmul(del_phi_1,f+np.matmul(g,u))
    sigma_2_bar = sigma_2/(np.matmul(sigma_2.T,sigma_2)+1)
    M = sigma_2/(np.matmul(sigma_2.T,sigma_2)+1)**2
    phi_1 = np.array([x_1**2,x_1*x_2,x_2**2]).T
    W_1_hat_dot = -a_1 * (M) * (np.matmul(sigma_2.T,W_1_hat)+cost)
    D_1 = np.matmul(np.matmul(np.matmul(np.matmul(del_phi_1,g),np.linalg.inv(R)),g.T),del_phi_1.T)
    W_2_hat_dot = -a_2*(np.matmul(F2,W_2_hat)-F1*np.matmul(sigma_2_bar.T,W_1_hat)-0.25*np.matmul(D_1,W_2_hat*(np.matmul(M.T,W_1_hat))))
    V_hat = np.matmul(W_1_hat.T, phi_1)
    print(">>>>>",V_hat)
    x_dot = f + np.dot(g, u)
    return_vec = np.concatenate((x_dot,W_1_hat_dot,W_2_hat_dot))
    return return_vec



# Define the initial conditions and parameters
x0 = np.array([1,1]) # initial state of x
W1_hat = 1*np.random.rand(3,1).ravel()
W2_hat = 1*np.random.rand(3,1).ravel()
# print(x0.shape,W1_hat.shape)
initial_value = np.concatenate((x0,W1_hat,W2_hat))
# Define the time vector for integration
t = np.linspace(0, 100, 10000)

# Solve the ODE system using odeint
x = odeint(x_dot, initial_value, t)
W1_data = x[:,2:5]
W2_data = x[:,5:8]
print(W1_data[0,:],'\n',W1_data[-1,:])
print(W2_data[0,:],'\n',W2_data[-1,:])
# Plot the solution
fig,axes = plt.subplots(3,1)
axes[0].plot(t, x[:,1:2])
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



