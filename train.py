import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from sklearn.decomposition import PCA
import random

from datasets import *


# compute kl loss (not use now)
def compute_kl(real, pred):
    # 1e-4为归一化项
    return torch.sum((torch.log(pred + 1e-4) - torch.log(real + 1e-4)) * pred)


def KL_Loss(x_fake, x_real, y_fake, y_real, col_types, col_idxes):
    kl = 0.0
    unique_labels = torch.unique(y_fake)
    for label in unique_labels:
        x_tem_fake = x_fake[torch.where(y_fake == label)]
        x_tem_real = x_real[torch.where(y_real == label)]
        for i in range(len(col_types)):
            sta = col_idxes[i][0]
            end = col_idxes[i][1]
            fakex = x_tem_fake[:, sta:end + 1]
            realx = x_tem_real[:, sta:end + 1]
            if col_types[i] == "gmm":
                fake2 = fakex[:, 1:]
                real2 = realx[:, 1:]
                # 相当于计算每一列的均值，然后比较不同均值之间的差异
                dist = torch.sum(fake2, dim=0)
                dist = dist / torch.sum(dist)
                real = torch.sum(real2, dim=0)
                real = real / torch.sum(real)
                # 比较两个均值的差异
                kl += compute_kl(real, dist)
            else:
                dist = torch.sum(fakex, dim=0)
                dist = dist / torch.sum(dist)

                real = torch.sum(realx, dim=0)
                real = real / torch.sum(real)
                kl += compute_kl(real, dist)

    return kl


# 练习使用条件GAN训练判别器和分类器
def CGAN_train(X, Y, G, D, dataset, epochs, lr_gen, lr_dis, z_dim, col_types, col_idxes, print_every=10,
               generate_every=50,
               num_gen=50, step_per_epoch=32, GPU=False):
    G.train()
    D.train()
    if GPU:
        G.cuda()
        D.cuda()
    # 设定的真实值，小于1是因为要增大判别器的训练难度
    real_label = 0.8
    fake_label = 0
    D_optim = optim.RMSprop(D.parameters(), lr=lr_dis)
    G_optim = optim.Adam(G.parameters(), lr=lr_gen)
    conditions = np.unique(Y)
    num_c = int(np.max(conditions)) + 1
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()
    if GPU:
        X = X.cuda()
        Y = Y.cuda()
    data_set = Data.TensorDataset(X, Y)
    loss = torch.nn.BCELoss()
    data_loader = Data.DataLoader(dataset=data_set, batch_size=step_per_epoch, shuffle=True)
    g_losses, d_losses = [], []
    for epoch in range(epochs):
        for i, (batch_X, batch_y) in enumerate(data_loader):
            batch_size = batch_X.shape[0]
            real_labels = torch.empty(batch_size, 1).fill_(real_label)
            fake_labels = torch.empty(batch_size, 1).fill_(fake_label)
            if GPU:
                real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()
            # 训练判别器
            noise = torch.randn(batch_size, z_dim)
            # 每次的标签都是随机生成的
            gen_c = torch.randint(0, num_c, (batch_size,))
            if GPU:
                noise, gen_c = noise.cuda(), gen_c.cuda()
            gen_X = G(noise, gen_c)
           # print('gen_x:{}'.format(gen_X.shape))
            D_optim.zero_grad()
            # print(batch_X.shape)
            # print(batch_y.shape)
            validity_real = D(batch_X, batch_y)
            d_real_loss = F.binary_cross_entropy(validity_real, real_labels)
            validity_gen = D(gen_X, gen_c)
          #  print(validity_gen)
            d_fake_loss = F.binary_cross_entropy(validity_gen, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            G_optim.zero_grad()
            D_optim.zero_grad()
            d_loss.backward()
            d_losses.extend([d_loss.data.item()] * batch_size)
            D_optim.step()
            # 训练生成器
            noise = torch.randn(batch_size, z_dim)
            gen_c = torch.randint(0, num_c, (batch_size,))
            if GPU:
                noise, gen_c = noise.cuda(), gen_c.cuda()
            gen_X = G(noise, gen_c)
            gen_label = D(gen_X, gen_c)
            #在这里，损失函数是为了让标签的分布更加靠近真实标签分布，
            #添加了一个KL散度的部分是为了让生成的数据各个属性上的均值和标准分布更加接近。
            g_loss = loss(gen_label, real_labels)
            kl_loss = KL_Loss(gen_X, X, gen_c, Y, col_types, col_idxes)
            g_loss += kl_loss
            g_loss.backward(retain_graph=True)
            g_losses.extend([g_loss.data.item()] * batch_size)
            G_optim.step()

        if epoch % generate_every == 0:
            G.eval()
            D.eval()
            noise = torch.randn(num_gen, z_dim)
            gen_c = torch.randint(0, num_c, (num_gen,))
            if GPU:
                noise, gen_c = noise.cuda(), gen_c.cuda()
            gen_X = G(noise, gen_c).cpu().detach().numpy()
            gen_c = gen_c.cpu().detach().numpy()
            gen_X = PCA(n_components=2, ).fit_transform(gen_X)
            plt.scatter(gen_X[:, 0], gen_X[:, 1], c=gen_c)
            plt.savefig('{}_{}'.format(dataset, epoch))
            plt.cla()
            G.train()
            D.train()

        if epoch % print_every == 0:
            print("epoch:{},g_loss:{},d_loss:{}".format(epoch, np.mean(g_losses), np.mean(g_losses)))


def CGAN_Generate(G, z_dim, c, num_gen, GPU):
    z = torch.randn(num_gen, z_dim)
    c = torch.empty(num_gen, ).fill_(c)
    if GPU:
        G = G.cuda()
        z = z.cuda()
        c = c.cuda()
    X = G(z, c).cpu().detach().numpy()
    return X
