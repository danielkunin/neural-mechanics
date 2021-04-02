import torch
from torch.utils.data import Dataset
import numpy as np

# Builds a random dataset such that when used in linear regression
# their is a certain SVD structure to the Hessian of the MSE loss
# n: number of data points
# d: dimension of data points
# angle: angle of major axis in degrees
# condition: condition number ratio of major and minor axis
# w: true weight vector
# b: true bias
# sigma: variance of error
def multivariate_dataset(n, d, condition, w, b, sigma=1.0):
    # Build Eigenvector matrix
    R = np.eye(d+1)
    # Build Eigenvector matrix
    Sigma = np.diag(np.linspace(np.sqrt(condition),1,d+1))
    # Build data matrix
    Q, _ = np.linalg.qr(np.random.randn(n, d+1))
    X = Q.dot(Sigma).dot(R)
    # Project data matrix so last column is all one
    v = X[:,-1] - np.linalg.norm(X[:,-1]) / np.sqrt(n) * np.ones(n)
    H = np.eye(n) - 2 * np.outer(v, v) / np.linalg.norm(v)**2
    X = H.dot(X) * np.sqrt(n) / np.linalg.norm(X[:,-1])
    X = X[:,:-1]
    # Sample Output from data matrix
    eps = np.random.normal(0,sigma,n)
    Y = X.dot(w) + b + eps
    return X, Y

class SyntheticRegression(Dataset):
    def __init__(self, n, d, condition=10, train=True, sigma=1.0):
        w = np.random.normal(0,1,d)
        b = 0.5
        train_frac = 0.7
        X, Y = multivariate_dataset(n, d, condition, w, b, sigma)
        if train:
            X = X[:train_frac*n]
            Y = Y[:train_frac*n]
        else:
            X = X[train_frac*n:]
            Y = Y[train_frac*n:]

        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
