import csv
import numpy as np
import matplotlib.pyplot as plt

temp_data = []
d = 17
n = 4

with open('uk.dat') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')

    first = True
    for row in reader:
        if (first):
            first = False
            continue
        temp_data.append([int(row[j]) for j in range(1, 5)])

data = np.array(temp_data)
print(data)

mean = np.zeros((17, 1))

for i in range(d):
    mean[i] = sum(data[i]) / 4


print("Mean")
print(mean)

X = np.array([data[i] - mean[i] for i in range(d)])

print("\nX")
print(X)

C = (1/(n-1)) * np.dot(X, np.transpose(X))

print("\nC")
print(C)

e, v = np.linalg.eig(C)

ev = [(e[i], v[:,i]) for i in range(17)]

ev = sorted(ev, key=lambda tup: tup[0], reverse=True)

print("\neigenvalues")
print(np.around(e, 2))

print("\neigenvectors")
print(np.around(v, 2))

largest_v = [ev[i] for i in range(2)]

k = 2

result = np.dot(np.array([t[1] for t in largest_v]), X)

labels=["England", "Wales", "Scotland", "N. Ireland"]
for i in range(4):
    x = result[:,i][0]
    y = result[:,i][1]
    plt.scatter(x,y,label=labels[i])

plt.legend()
plt.show()

lostVariance = sum([ev[i][0] for i in range(2)]) / sum(t[0] for t in ev)
print(lostVariance)
