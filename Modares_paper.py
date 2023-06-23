import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import block_diag

from scipy import linalg as la


A = np.array([[-1,2],[2.2,1.7]])
B = np.array([[2],[1.6]])
C = np.array([[1,2]])
F = np.array([[-1]])
Q = 6*np.eye(3)
R = np.eye(1)
gamma = 0.8

print(A.shape,B.shape,C.shape,F.shape)

A_dash = block_diag(A,F)
B_dash = np.concatenate((B,np.zeros((1,1))),axis=0)
print(A_dash,'\n',B_dash)


num_iterations = 20

P = np.zeros((num_iterations,3,3))
P[0,:,:] = []
for iter in num_iterations:
    P[iter+1,:,:] = 