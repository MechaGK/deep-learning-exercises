import numpy as np
import matplotlib.pyplot as mpl
import csv

xs = np.array([[0,1],[1,1],[1,0],[1,2],[2,3],[5,3],[2,5],[4,3]])

mean = np.array([0,0])
for x in xs:
    mean = np.add(mean, x)

n = len(xs)

mean = np.multiply(1/n, mean)

print("Mean")
print(mean)

X = np.array([ np.subtract(x, mean) for x in xs])

print("\nX")
print(X)

C = 1/(n-1) * np.dot(np.transpose(X), X)

print("\nC")
print(C)

e, v = np.linalg.eig(C)

print("\neigenvalues")
print(e)

print("\neigenvectors")
print(v)

print("\n")

proj0 = np.dot(np.transpose(np.transpose(v)[0]), np.transpose(X))
mpl.scatter(proj0[0], proj0[1])

proj1 = np.dot(np.transpose(v), np.transpose(X))
mpl.scatter(proj1[0], proj1[1])
mpl.show()





