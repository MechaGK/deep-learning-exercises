import pandas
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

df = pandas.read_csv('breastcancer.csv', sep=',', header=0, index_col=[0, 1],
                       converters={'Class': lambda s: 0 if s == "benign" else 1}, na_values='NA')
df = df.dropna()


pca_df = df.drop('Class', axis=1)

mean = pca_df.mean().get_values()

centered = pca_df - mean

cov = (1/(len(centered.get("Cl.thickness").get_values())-1)) * np.dot(np.transpose(centered), centered)

eigenvalues, eigenvectors = np.linalg.eig(cov)

idx = np.flip(eigenvalues.argsort())
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

print("\neigenvalues")
print(np.around(eigenvalues, 2))

print("\neigenvectors")
print(np.around(eigenvectors, 2))

E = eigenvectors[:,:2]

result = np.dot(np.transpose(E), np.transpose(centered))
result.shape


fig, ax = plt.subplots()

for i in range(683):
    ax.scatter(result[:,i][0], result[:,i][1], color=('blue' if df.get('Class').iat[i] else 'red'))

# plt.show()


model = sm.Logit(df.get('Class'), sm.add_constant(pca_df))

res = model.fit()

res.summary()
pred = res.pred_table()
precision = pred[1,1] / (pred[1,1] + pred[0,1])
recall = pred[1,1] / (pred[1,1] + pred[1,0])

F = 2 * (precision * recall) / (precision + recall)

total_F = F


def f(x, params):
    return sum([np.power(x, i) * params[i] for i in range(len(params))])


dx = np.linspace(-15, 5, 200)
plt.plot(dx, f(dx, res.params))
plt.ylim(-20, 20)
plt.show()

tprs = []
for i in range(100):
    t = res.pred_table((i+1)/100)
    print(t)
    tprs.append(t[0,0]/(t[0,0]+t[1,0]))

fprs = []
for i in range(100):
    t = res.pred_table((i+1)/100)
    fprs.append(t[0,1]/(t[1,1]+t[0,1]))

plt.plot(fprs, tprs)
plt.ylim(0, 1)
plt.xlim(0, 1)
# plt.show()

Fs = []

steps = 683 // 10
for i in range(10):
    data = df.drop(df.index[list(range(i * steps, (i+1) * steps))])
    test = df.iloc[(i * steps):((i+1) * steps)]

    model = sm.Logit(data.get('Class'), sm.add_constant(data.drop('Class', axis=1)))

    res = model.fit()

    res.summary()
    pred = res.pred_table()
    precision = pred[1,1] / (pred[1,1] + pred[0,1])
    recall = pred[1,1] / (pred[1,1] + pred[1,0])

    F = 2 * (precision * recall) / (precision + recall)

    Fs.append(F)

print(sum(Fs)/len(Fs))
print(total_F)
