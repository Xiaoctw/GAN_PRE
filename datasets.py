import os
import numpy as np
import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import json

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from pathlib import Path


def LoadParams(dataset):
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
    Y = np.concatenate([Y1, Y2]).reshape(-1, )
    return X, Y


def Wine():
    path = Path(__file__).parent / 'data' / ('wine.csv')
    df = pd.read_csv(path)
    matrix = df.values
    X, Y = matrix[:, :-1], matrix[:, -1]
    X = 2 * X - 1
    Y = LabelEncoder().fit_transform(Y)
    return X, Y


def Yeast():
    file_name = './data/yeast.csv'
    wild_data = pd.read_csv(file_name)
    data = []
    # 连续型属性
    num_data = wild_data[['0', '1', '2', '3', '6', '7']].values
    data.append(num_data)
    Y = wild_data['8'].values.astype(np.int)
    col_types = ["ordinal", "ordinal", "ordinal", "ordinal", "ordinal", "ordinal"]
    col_idxes = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
    tem = 7
    # cat_data = data[['4', '5']].values
    # 离散型属性
    cat_list = ['4', '5']
    for feat in cat_list:
        cat_feat, num_one_hot = one_hot_feature(wild_data[feat])
        col_types.append('one-hot')
        col_idxes.append([tem, tem + num_one_hot-1])
        tem=tem+num_one_hot
        data.append(cat_feat)
    X = np.concatenate(data, axis=1)
    # print('col_types:{}'.format(col_types))
    # print("col_idxes:{}".format(col_idxes))
    # print('x_dim:{}'.format(X.shape[1]))
    # print('c_dim:{}'.format(np.unique(Y).shape[0]))
    return X, Y


def one_hot_feature(arr):
    unique_list = np.unique(arr)
    classes_dict = {c: np.identity(len(unique_list))[i, :] for i, c in  # identity创建方矩阵
                    enumerate(unique_list)}  # 字典 key为label的值，value为矩阵的每一行
    # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列
    labels_onehot = np.array(list(map(classes_dict.get, arr)),  # get函数得到字典key对应的value
                             dtype=np.int32)
    return labels_onehot, unique_list.shape[0]


if __name__ == '__main__':
    X, Y = Yeast()
    print(X.shape)
    print(Y.shape)
