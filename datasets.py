import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

def RandomDataset():
    mean1 = [2, 3]
    mean2 = [6, 7]
    cov1 = [[2, 1], [1, 2]]
    cov2 = [[0.7, 1], [1, 0.7]]
    X1 = np.random.multivariate_normal(mean1, cov1, 100)
    X2 = np.random.multivariate_normal(mean2, cov2, 40)
    Y1 = np.ones(100)
    Y2 = np.zeros(40)
    X = np.concatenate([X1, X2])
    Y = np.concatenate([Y1, Y2]).reshape(-1,1)
    return np.concatenate([X,Y],axis=1)