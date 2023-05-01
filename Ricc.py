import numpy as np
from scipy.linalg import solve_continuous_are

# Define the system matrices
# a = np.array([[0, 1], [-1, -1]])
# b = np.array([[0], [1]])

A = np.array([[-1.01887, 0.90506, -0.00215],[0.82225, -1.07741, -0.17555],[0,0,-1]]) # parameter a
B = np.array([[0],[0],[1]]) # parameter b

# Define the cost matrices
Q = np.eye(3)
R = np.array([[1]])

# Solve the ARE
k = solve_continuous_are(A, B, Q, R)

print("The state feedback gain matrix K is:\n", k)
