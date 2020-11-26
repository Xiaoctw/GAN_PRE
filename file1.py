import os
import json
import numpy as np
import warnings
import torch
from datasets import *
from models import *
from train import *
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

mean1 = [2, 3]
mean2 = [6, 7]
cov1 = [[2, 1], [1, 2]]
cov2 = [[0.7, 1], [1, 0.7]]


def plot_Data(X):
    plt.scatter(X[:, 0], X[:, 1], c=X[:, 2])
    plt.show()


def LoadParams(dataset, cond=False):
    if cond:
        path = Path(__file__).parent/'params' / ('param_' + dataset+'_random.json')
    else:
        path = Path(__file__).parent / 'params' / ('param_' + dataset + '.json')
    f=open(path,'r')
    params = json.load(f)
    return params


if __name__ == '__main__':
    GPU=torch.cuda.is_available()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    X = RandomDataset()
    params = LoadParams('random')
    G = VGAN_generator(z_dim=params["z_dim"], hidden_dim=params['hidden_dim'], x_dim=params['x_dim'], num_layer=params['num_layer'],
                       col_types=params['col_types'],
                       col_idxes=params['col_idxes'], condition=params['condition'], c_dim=0)
    D = VGAN_discriminator(hidden_dim=params['hidden_dim'], x_dim=params['x_dim'], num_layer=params['num_layer']
                       ,condition=params['condition'], c_dim=0)
    VGAN_train(X, G, D, epochs=256, lr=3e-5,
               z_dim=8, print_every=10, GPU=GPU)
    X1=VGAN_Generate(G,z_dim=params["z_dim"],num_generate=40,GPU=GPU)
    plot_Data(X)
    plot_Data(X1)
