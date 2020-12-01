import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import random

from datasets import *


def VGAN_train(X, G, D, epochs, lr_gen, lr_dis,
               z_dim, plot_Data, print_every=10, generate_every=50, num_gen=50, GPU=False):
    if GPU:
        G.cuda()
        D.cuda()
    D_optim = optim.RMSprop(D.parameters(), lr=lr_dis)
    G_optim = optim.RMSprop(G.parameters(), lr=lr_gen)
    # data_set = Data.TensorDataset(X)
    X = torch.from_numpy(X).float()
    dataloader = Data.DataLoader(dataset=X, batch_size=X.shape[0] // 10, shuffle=True)  # num_workers=-1)
    for step in range(epochs):
        D_losses, G_losses = [], []
        for i, x_real in enumerate(dataloader):
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
            real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
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
            if i % 2 == 0:
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
                G_losses.append(G_Loss.item())
            D_losses.append(D_Loss.item())
        if (step + 1) % print_every == 0:
            print('iterator {}, D_Loss:{}, G_Loss:{}'.format((step + 1), np.mean(D_losses), np.mean(G_losses)))
        if (step + 1) % generate_every == 0:
            X1 = VGAN_Generate(G, z_dim=z_dim, num_generate=num_gen, GPU=GPU)
            plot_Data(X1, 'generated')
    return G, D


def VGAN_Generate(G, z_dim, num_generate, GPU):
    z = torch.randn(num_generate, z_dim)
    if GPU:
        G = G.cuda()
        z = z.cuda()
    X = G(z).cpu().detach().numpy()
    X[:, -1] = (X[:, -1] > 0.5).astype(np.int)
    # for i in range(num_generate):
    #     z = torch.randn(1, z_dim)
    #     if GPU:
    #         z = z.cuda()
    #     x: torch.Tensor = G(z)
    #     X.append(x.numpy())
    # X = np.concatenate(X)
    return X


# 练习使用条件GAN训练判别器和分类器
def CGAN_train(X, Y, G, D, epochs, lr_gen, lr_dis, z_dim, print_every=10, generate_every=50, c_dim=2,
               num_gen=50, step_per_epoch=32, GPU=False):
    if GPU:
        G.cuda()
        D.cuda()
    D_optim = optim.RMSprop(D.parameters(), lr=lr_dis)
    G_optim = optim.RMSprop(G.parameters(), lr=lr_gen)
    conditions = np.unique(Y)
    # data_set = Data.TensorDataset(X)
    data_set = DataSet(X, Y)
    for epoch in range(epochs):
        # for step, (batch_x, batch_y) in enumerate(data_loader):
        #     batch_y=batch_y.view(-1,1)
        #     #获得了改组数据的one-hot标签
        #     c_real=torch.zeros(batch_y.shape[0],z_dim).scatter_(1, batch_y, 1)
        #     if GPU:
        D_losses = []
        G_losses = []
        for it in range(len(data_set) // step_per_epoch):
            # 随机选择一个类
            c = int(np.random.choice(conditions))
            x_real = data_set.random_choice(label=c, number=step_per_epoch)
            c_real = np.zeros((step_per_epoch, c_dim))
            c_real[:, c] = 1
            x_real = torch.from_numpy(x_real).float()
            c_real = torch.from_numpy(c_real).float()
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                x_real = x_real.cuda()
                c_real = c_real.cuda()
                z = z.cuda()
            x_fake = G(z, c_real)
            y_real = D(x_real, c_real)
            y_fake = D(x_fake, c_real)
            fake_label = torch.zeros(y_fake.shape[0], 1).float()
            real_label=torch.ones(y_real.shape[0],1).float()
            #real_label = torch.from_numpy(real_label).float()
            if GPU:
                fake_label = fake_label.cuda()
                real_label = real_label.cuda()
            label=torch.cat((fake_label,real_label),0)
            y=torch.cat((y_real,y_fake),0)
            # D_loss1 = F.binary_cross_entropy(y_real, real_label)
            # D_loss2 = F.binary_cross_entropy(y_fake, fake_label)
            D_loss = F.binary_cross_entropy(y,label)
            G_optim.zero_grad()
            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()
            D_losses.extend([D_loss.item()] * step_per_epoch)
            if (epoch + 1) % 5 == 0:
                # 控制生成器训练次数
                # 开始训练生成器
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                x_fake = G(z, c_real)
                y_fake = D(x_fake, c_real)
                real_label = torch.ones(y_fake.shape[0], 1)
                if GPU:
                    real_label = real_label.cuda()
                G_Loss = F.binary_cross_entropy(y_fake, real_label)
                G_optim.zero_grad()
                D_optim.zero_grad()
                G_Loss.backward()
                G_losses.extend([G_Loss.item()] * step_per_epoch)
                G_optim.step()

        if (epoch + 1) % print_every == 0:
            tensor_X = torch.from_numpy(X).float()
            #  Y = torch.from_numpy(Y).float()
            tensor_c = np.zeros((tensor_X.shape[0], 2))
            tensor_c[np.where(Y == 1), 1] = 1
            tensor_c[np.where(Y == 0), 0] = 1
            tensor_c = torch.from_numpy(tensor_c).float()
            if GPU:
                tensor_X = tensor_X.cuda()
                tensor_c = tensor_c.cuda()
            pred_label = np.array(D(tensor_X, tensor_c) > 0.5)
            real_label = np.ones([tensor_c.shape[0], 1])
            accuracy = np.sum(pred_label == real_label) / tensor_c.shape[0]
            print('iterator {}, D_Loss:{}, G_Loss:{},accuracy:{}'.format((epoch + 1), np.mean(D_losses),
                                                                         np.mean(G_losses), accuracy))
        if (epoch + 1) % generate_every == 0:
            X, Y = [], []
            for _ in range(num_gen // 2):
                c = int(np.random.choice(conditions))
                X1 = CGAN_Generate(G, z_dim=z_dim, c_dim=c_dim, c=c, num_gen=2, GPU=GPU)
                X.append(X1)
                Y.extend([c] * 2)
            X = np.concatenate(X)
            plt.scatter(X[:, 0], X[:, 1], c=Y)
            plt.show()


def CGAN_Generate(G, z_dim, c_dim, c, num_gen, GPU):
    z = torch.randn(num_gen, z_dim)
    c_pred = torch.zeros(num_gen, c_dim)
    c_pred[0][c] = 1
    if GPU:
        G = G.cuda()
        z = z.cuda()
        c_pred = c_pred.cuda()
    X = G(z, c_pred).cpu().detach().numpy()
    # for i in range(num_generate):
    #     z = torch.randn(1, z_dim)
    #     if GPU:
    #         z = z.cuda()
    #     x: torch.Tensor = G(z)
    #     X.append(x.numpy())
    # X = np.concatenate(X)
    return X
