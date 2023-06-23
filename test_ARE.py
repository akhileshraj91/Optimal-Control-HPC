import numpy as np
from scipy import linalg as la

a = np.array([[0, 1], [0, -1]])
b = np.array([[1, 0], [2, 1]])
# q = np.array([[-4, -4], [-4, 7]])
# r = np.array([[9, 3], [3, 1]])
q = np.eye(2)
r = np.eye(2)

x = la.solve_discrete_are(a, b, q, r)

print(x)

R = la.solve(r + b.T.dot(x).dot(b), b.T.dot(x).dot(a))

print(np.allclose(a.T.dot(x).dot(a) - x - a.T.dot(x).dot(b).dot(R), -q))
