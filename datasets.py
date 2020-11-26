import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

def RandomDataset(num_positive,num_negative):

    mean1 = [2, 3]
    mean2 = [6, 7]
    cov1 = [[2, 1], [1, 2]]
    cov2 = [[0.7, 1], [1, 0.7]]
    X1 = np.random.multivariate_normal(mean1, cov1, num_positive)
    X2 = np.random.multivariate_normal(mean2, cov2, num_negative)
    Y1 = np.ones(num_positive)
    Y2 = np.zeros(num_negative)
    X = np.concatenate([X1, X2])
    Y = np.concatenate([Y1, Y2]).reshape(-1,1)
    return np.concatenate([X,Y],axis=1)