import torch
import torch.nn as nn
import torch.nn.functional as fun

import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.autograd import Variable


class VGAN_generator(nn.Module):
    """
        The generator for vanilla GAN.
        It takes as input a Gaussian noise z and a condition vector c (optional),
          and produces a "fake" vector x.
        To this end, it employs multiple fully-connected layers with batch normalization.
        """

    def __init__(self, z_dim, hidden_dim, x_dim, num_layer, col_types, col_idxes, condition=False, c_dim=0):
        super(VGAN_generator, self).__init__()
        # 先进行batch_norm
        self.input = nn.Linear(z_dim + c_dim, hidden_dim)
        self.inputbn = nn.BatchNorm1d(hidden_dim)
        self.hidden = []
        self.BN = []
        self.col_types = col_types
        self.col_idxes = col_idxes
        self.x_dim = x_dim  # 生成数据的维度
        self.c_dim = c_dim  # 条件向量的维度
        self.condition = condition
        self.num_layer = num_layer
        for i in range(num_layer):
            fc = nn.Linear(hidden_dim, hidden_dim)
            setattr(self, "fc%d" % i, fc)
            self.hidden.append(fc)
            bn = nn.BatchNorm1d(hidden_dim)
            setattr(self, "bn%d" % i, bn)
            self.BN.append(bn)
        self.output = nn.Linear(hidden_dim, x_dim)
        self.outputbn = nn.BatchNorm1d(x_dim)

    def forward(self, z, c=None):
        if self.condition:
            assert c is not None
            z = torch.cat((z, c), 1)
        z = self.input(z)
        z = fun.relu(self.inputbn(z))
        for i in range(self.num_layer):
            z = self.hidden[i](z)
            z = self.BN[i](z)
            z = fun.relu(z)
        x = self.output(z)
        x = self.outputbn(x)
        output = []
        for i in range(len(self.col_types)):
            start = self.col_idxes[i][0]
            end = self.col_idxes[i][1]
            if self.col_types[i] == 'binary':
                temp = fun.sigmoid(x[:, start:end+1])
            elif self.col_types[i] == 'normalize':
                # 数据转化到了[-1,1]之间
                temp = fun.tanh(x[:, start:end+1])
            elif self.col_types[i] == 'one-hot':
                temp = torch.softmax(x[:, start:end+1], dim=1)
            elif self.col_types[i] == 'gmm':
                temp1 = torch.tanh(x[:, start:start + 1])
                temp2 = torch.softmax(x[:, start + 1:end+1], dim=1)
                temp = torch.cat((temp1, temp2), dim=1)
            else:
                # self.col_type[i] == 'ordinal':
                temp = torch.sigmoid(x[:, start:end+1])
            output.append(temp)
        output = torch.cat(output, dim=1)
        return output


class VGAN_discriminator(nn.Module):
    """
        The discriminator for vanilla GAN.
        It takes as input the real/fake data,
          and uses an MLP to produce label (1: real; 0: fake)
        """
    def __init__(self, x_dim, hidden_dim, num_layer, condition=False, c_dim=0, wgan=False):
        super(VGAN_discriminator, self).__init__()
        self.num_layer = num_layer
        self.input = nn.Linear(x_dim + c_dim, hidden_dim)
        self.hidden = []
        self.Dropout = nn.Dropout(p=0.5)
        self.condition = condition
        self.wgan=wgan
        for i in range(num_layer):
            fc = nn.Linear(hidden_dim, hidden_dim)
            setattr(self, "fc%d" % i, fc)
            self.hidden.append(fc)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self,z,c=None):
        if self.condition:
            assert c is not None
            z = torch.cat((z, c), dim=1)
        z = self.input(z)
        z = torch.relu(z)
        z = self.Dropout(z)
        for i in range(len(self.hidden)):
            z = self.hidden[i](z)
            z = torch.relu(z)
            z = self.Dropout(z)
        z = self.output(z)
        if self.wgan:
            return z
        else:
            return torch.sigmoid(z)


