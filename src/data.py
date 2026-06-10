
import numpy as np
import urllib.request
import os

def load_california_housing(path="../datasets/housing.csv"):
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    data_path = path
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
    urllib.request.urlretrieve(url, data_path)
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1, usecols=range(9))
    n_data = data[~np.isnan(data).any(axis=1)]
    X = n_data[:, :-1]
    Y = n_data[:, -1:]
    return X, Y

def train_test_split(X, Y, test_size=0.2):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1e-8
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)