import os
import numpy as np
import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import json

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from pathlib import Path


def LoadParams(dataset, cond=False):
    if cond:
        path = Path(__file__).parent / 'params' / ('param_' + dataset + '_condition.json')
    else:
        path = Path(__file__).parent / 'params' / ('param_' + dataset + '.json')
    f = open(path, 'r')
    params = json.load(f)
    return params


class DataSet:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        labels = np.unique(Y)
        self.label2idx = {}
        for label in labels:
            # 获得对应标签的索引序列
            self.label2idx[label] = np.where(Y == label)[0]

    def random_choice(self, label, number):
        assert label in self.label2idx
        idx = np.random.choice(self.label2idx[label], number)
        return self.X[idx].reshape(number, -1)

    def __len__(self):
        return self.X.shape[0]


def RandomDataset(num_positive, num_negative):
    mean1 = [2, 3]
    mean2 = [6, 7]
    cov1 = [[2, 1], [1, 2]]
    cov2 = [[0.7, 1], [1, 0.7]]
    X1 = np.random.multivariate_normal(mean1, cov1, num_positive)
    X2 = np.random.multivariate_normal(mean2, cov2, num_negative)
    X3 = np.random.multivariate_normal([-10, -10], cov2, 2)
    Y1 = np.ones(num_positive)
    Y2 = np.zeros(num_negative + 2)
    X = np.concatenate([X1, X2, X3])
    X = 2 * MinMaxScaler().fit_transform(X) - 1
    Y = np.concatenate([Y1, Y2]).reshape(-1, 1)
    return np.concatenate([X, Y], axis=1)


def RandomDatasetCondition(num_positive, num_negative):
    mean1 = [2, 3]
    mean2 = [2, 10]
    cov1 = [[2, 1], [1, 2]]
    cov2 = [[0.7, 1], [1, 0.7]]
    X1 = np.random.multivariate_normal(mean1, cov1, num_positive)
    X2 = np.random.multivariate_normal(mean2, cov2, num_negative)
    Y1 = np.ones(num_positive)
    Y2 = np.zeros(num_negative)
    X = np.concatenate([X1, X2])
    X = (2 * MinMaxScaler().fit_transform(X)) / 2
    Y = np.concatenate([Y1, Y2]).reshape(-1, 1)
    return X, Y


def Wine():
    path = Path(__file__).parent / 'data' / ('wine.csv')
    df = pd.read_csv(path)
    matrix = df.values
    X, Y = matrix[:, :-1], matrix[:, -1]
    X=2*X-1
    Y = LabelEncoder().fit_transform(Y)
    return X, Y
