import os
import json
import numpy as np
import warnings
import torch
import argparse
from datasets import *
from models import *
from train import *
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs to train.')
parser.add_argument('--lr_G', default=5e-5)
parser.add_argument('--lr_D', default=3e-4)
parser.add_argument('--generate_every', default=100)
parser.add_argument('--train_type', default='WGAN')
parser.add_argument('--print_every', default=20)
parser.add_argument('--dataset', default='Wine', choices=['random_condition', 'Yeast', 'Wine'])
args = parser.parse_args()

num_gen = 200
lr_gen = args.lr_G
lr_dis = args.lr_D
epochs = args.epochs
print_every = args.print_every
generate_every = args.generate_every
step_per_epoch = 32
random.seed(1)
torch.manual_seed(1)
dataset = args.dataset
X, Y = load_dataset(dataset)

if __name__ == '__main__':
    plt.switch_backend('agg')
    GPU = torch.cuda.is_available()
    print('GPU:{}'.format(GPU))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    X1 = PCA(n_components=2, ).fit_transform(X)
    plt.scatter(X1[:, 0], X1[:, 1], c=Y)
    plt.savefig('pictures/init_{}.png'.format(dataset))
    # plt.show()
    plt.cla()
    params = LoadParams(dataset)
    G = VCGAN_generator(z_dim=params["z_dim"], hidden_dim=params['hidden_dim'], x_dim=params['x_dim'],
                        num_layer=params['num_layer'],
                        col_types=params['col_types'],
                        col_idxes=params['col_idxes'], c_dim=params['c_dim'])
    D = VCGAN_discriminator(hidden_dim=params['hidden_dim'], x_dim=params['x_dim'], num_layer=params['num_layer']
                            , c_dim=params['c_dim'],wgan=True)
    G.apply(init_weights)
    D.apply(init_weights)

    WGAN_train(X, Y, G, D, dataset=dataset, epochs=epochs, lr_gen=lr_gen, lr_dis=lr_dis,
               col_types=params['col_types'],
               col_idxes=params['col_idxes'],
               z_dim=params['z_dim'], print_every=print_every,
               generate_every=generate_every,
               num_gen=num_gen, step_per_epoch=step_per_epoch, GPU=GPU,
               )

    G = G.cpu()
    torch.save(G, 'models/G_{}_{}_{}.pkl'.format(dataset, epochs, args.train_type))
    CGAN_Generate(G, z_dim=params['z_dim'], num_c=params['c_dim'], dataset=dataset, num_gen=num_gen)
    # plt.scatter(X[:, 0], X[:, 1], c=Y)
    # plt.show()
