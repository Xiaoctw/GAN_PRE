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

num1 = 400
num2 = 100
num_gen = 200
lr_gen = 1e-5
lr_dis = 3e-4
epochs = 500
print_every = 10
generate_every = 40
step_per_epoch = 32

if __name__ == '__main__':
    plt.switch_backend('agg')
    GPU = torch.cuda.is_available()
    print('GPU:{}'.format(GPU))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    X, Y = RandomDatasetCondition(num1, num2)
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.show()
    params = LoadParams('random_condition')
    G = VGAN_generator(z_dim=params["z_dim"], hidden_dim=params['hidden_dim'], x_dim=params['x_dim'],
                       num_layer=params['num_layer'],
                       col_types=params['col_types'],
                       col_idxes=params['col_idxes'], c_dim=params['c_dim'])
    D = VGAN_discriminator(hidden_dim=params['hidden_dim'], x_dim=params['x_dim'], num_layer=params['num_layer']
                           , c_dim=params['c_dim'])
    CGAN_train(X, Y, G, D, epochs=epochs, lr_gen=lr_gen, lr_dis=lr_dis, z_dim=params['z_dim'], print_every=print_every,
               generate_every=generate_every, c_dim=params['c_dim'],
               num_gen=num_gen, step_per_epoch=step_per_epoch, GPU=GPU,
               )
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()
