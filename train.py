import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
import math


def VGAN_train(X, G, D, epochs, lr,
               z_dim, print_every=10, GPU=False):
    if GPU:
        G.cuda()
        D.cuda()
    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.00001)
    # data_set = Data.TensorDataset(X)
    X = torch.from_numpy(X).float()
    dataloader = Data.DataLoader(dataset=X, batch_size=X.shape[0] // 10, shuffle=True)  # num_workers=-1)
    for step in range(epochs):
        D_losses, G_losses = [], []
        for x_real in dataloader:
            # 训练判别器
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                x_real = x_real.cuda()
                z = z.cuda()
            x_fake = G(z)
            y_real = D(x_real)
            y_fake = D(x_fake)
            # D_Loss = -(torch.mean(y_real) - torch.mean(y_fake)) # wgan loss
            fake_label = torch.zeros(y_fake.shape[0], 1)  # shape (y_fake,1)
            real_label = np.ones([y_real.shape[0], 1])  # shape (y_fake,1)
            # Avoid the suppress of Discriminator over Generator
            # 增加判别器的生成难度
            # random.uniform 从一个均匀分布[low, high)中随机采样
            real_label = real_label * 0.6 + np.random.uniform(0, 0.4, real_label.shape)
            real_label = torch.from_numpy(real_label).float()
            if GPU:
                fake_label = fake_label.cuda()
                real_label = real_label.cuda()
            D_Loss1 = F.binary_cross_entropy(y_real, real_label)
            D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
            D_Loss = D_Loss1 + D_Loss2
            G_optim.zero_grad()
            D_optim.zero_grad()
            D_Loss.backward()
            D_optim.step()
            # 训练生成器
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()

            x_fake = G(z)
            y_fake = D(x_fake)

            real_label = torch.ones(y_fake.shape[0], 1)
            if GPU:
                real_label = real_label.cuda()
            G_Loss = F.binary_cross_entropy(y_fake, real_label)
            G_optim.zero_grad()
            D_optim.zero_grad()
            G_Loss.backward()
            G_optim.step()
            D_losses.append(D_Loss.item())
            G_losses.append(G_Loss.item())
        if (step + 1) % print_every == 0:
            print('iterator {}, D_Loss:{}, G_Loss:{}'.format((step + 1), np.mean(D_losses), np.mean(G_losses)))
    return G, D


def VGAN_Generate(G, z_dim, num_generate, GPU):
    z = torch.randn(num_generate, z_dim)
    if GPU:
        G = G.cuda()
        z=z.cuda()
    z = torch.randn(num_generate, z_dim)
    X = G(z).detach().numpy()
    X[:,-1]=(X[:,-1]>0.5).astype(np.int)
    # for i in range(num_generate):
    #     z = torch.randn(1, z_dim)
    #     if GPU:
    #         z = z.cuda()
    #     x: torch.Tensor = G(z)
    #     X.append(x.numpy())
    # X = np.concatenate(X)
    return X
