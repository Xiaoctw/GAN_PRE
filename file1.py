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


def plot_Data(X, name):
    plt.scatter(X[:, 0], X[:, 1], c=X[:, 2])
    plt.savefig(name + '.png')
    plt.show()
    # 清除


num1 = 400
num2 = 100
num_gen = 200
lr_gen = 1e-4
lr_dis = 3e-4
epochs = 800
print_every = 10
generate_every = 100

if __name__ == '__main__':
    GPU = torch.cuda.is_available()
    print('GPU:{}'.format(GPU))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    X = RandomDataset(num1, num2)
    params = LoadParams('random')
    G = VGAN_generator(z_dim=params["z_dim"], hidden_dim=params['hidden_dim'], x_dim=params['x_dim'],
                       num_layer=params['num_layer'],
                       col_types=params['col_types'],
                       col_idxes=params['col_idxes'], c_dim=0)
    D = VGAN_discriminator(hidden_dim=params['hidden_dim'], x_dim=params['x_dim'], num_layer=params['num_layer']
                           , c_dim=0)
    VGAN_train(X, G, D, epochs=epochs, lr_gen=lr_gen, lr_dis=lr_dis,plot_Data=plot_Data,
               z_dim=params["z_dim"], print_every=print_every, generate_every=generate_every,num_gen=num_gen, GPU=GPU)
    X1 = VGAN_Generate(G, z_dim=params["z_dim"], num_generate=num_gen, GPU=GPU)
    plot_Data(X, 'original')
    #plot_Data(X1, 'generated')
