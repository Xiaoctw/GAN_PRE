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
parser.add_argument('--epochs',default=1000,type=int,help='Number of epochs to train.')
parser.add_argument('--dataset', default='random_condition', choices=['random_condition', 'Yeast', 'Wine'])
parser.add_argument('--num_gen',default=200,help='Number of generation')
args=parser.parse_args()


epochs = args.epochs
dataset = args.dataset
num_gen=args.num_gen
params = LoadParams(dataset)
model=torch.load('models/G_{}_{}.pkl'.format(dataset,epochs))
if __name__ == '__main__':
    CGAN_Generate(model, z_dim=params['z_dim'], num_c=params['c_dim'], dataset=dataset, num_gen=num_gen)
