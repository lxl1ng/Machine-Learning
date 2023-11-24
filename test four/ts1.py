import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.txt")
data = np.array(data)


def pca(data, k):
    # decentralization
    data = data - np.mean(data, axis=0)
    # calculate covariance matrix of X
    data_cov = np.cov(data, rowvar=False)
    # eigenvalue decomposition of the covariance matrix
    eigvalues, eigvectors = np.linalg.eig(data_cov)
    # pick first-k eigvector as W
    max_eigvalue_index = np.argsort(-eigvalues)[:k]
    W = eigvectors[:, max_eigvalue_index]

    Z = data @ W
    return Z


Z = pca(data, 2)
X = []
Y = []
i = 0
for z in Z:
    X.append(z[0])
    Y.append(z[1])
np.savetxt('data_to2.txt', Z, fmt='%f')
plt.scatter(X, Y, c='r', s=1)
plt.show()
