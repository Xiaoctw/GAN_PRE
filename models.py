import torch
import torch.nn as nn
import torch.nn.functional as F


class VGAN_generator(nn.Module):
    """
        The generator for vanilla GAN.
        It takes as input a Gaussian noise z and a condition vector c (optional),
          and produces a "fake" vector x.
        To this end, it employs multiple fully-connected layers with batch normalization.
        """

    def __init__(self, z_dim, hidden_dim, x_dim, num_layer, col_types, col_idxes, c_dim=0,
                 ):
        super(VGAN_generator, self).__init__()
        self.emb = nn.Embedding(c_dim, c_dim)
        self.input = nn.Linear(z_dim + c_dim, hidden_dim)
        self.inputbn = nn.BatchNorm1d(hidden_dim)
        self.col_types = col_types
        self.col_idxes = col_idxes
        self.x_dim = x_dim  # 生成数据的维度
        self.num_layer = num_layer
        for i in range(num_layer):
            fc = nn.Linear(hidden_dim, hidden_dim)
            setattr(self, "fc%d" % i, fc)
            bn = nn.BatchNorm1d(hidden_dim)
            setattr(self, "bn%d" % i, bn)
            setattr(self, 'relu{}'.format(i), nn.LeakyReLU(0.2, inplace=True))
        self.output = nn.Linear(hidden_dim, x_dim)
        self.outputbn = nn.BatchNorm1d(x_dim)

    def forward(self, z, c):
        z = torch.cat((z, self.emb(c)), -1)
        z = self.input(z)
        z = self.inputbn(z)
        z = F.leaky_relu(z,negative_slope=0.2)
        for i in range(self.num_layer):
            z = getattr(self, 'fc{}'.format(i))(z)
            z = getattr(self, 'bn{}'.format(i))(z)
            z = getattr(self, 'relu{}'.format(i))(z)
        x = self.output(z)
        output = []
        for i in range(len(self.col_types)):
            start = self.col_idxes[i][0]
            end = self.col_idxes[i][1]
            if self.col_types[i] == 'binary':
                temp = F.sigmoid(x[:, start:end + 1])
            elif self.col_types[i] == 'normalize':
                # 数据转化到了[-1,1]之间
                temp = F.tanh(x[:, start:end + 1])
            elif self.col_types[i] == 'one-hot':
                temp = torch.softmax(x[:, start:end + 1], dim=1)
            elif self.col_types[i] == 'gmm':
                temp1 = torch.tanh(x[:, start:start + 1])
                temp2 = torch.softmax(x[:, start + 1:end + 1], dim=1)
                temp = torch.cat((temp1, temp2), dim=1)
            else:
                # self.col_type[i] == 'ordinal':
                temp = torch.tanh(x[:, start:end + 1])
            output.append(temp)
        output = torch.cat(output, dim=1)
        return output


class VGAN_discriminator(nn.Module):
    """
        The discriminator for vanilla GAN.
        It takes as input the real/fake data,
          and uses an MLP to produce label (1: real; 0: fake)
        """

    def __init__(self, x_dim, hidden_dim, num_layer, c_dim=0, wgan=False, dropout=0.5):
        super(VGAN_discriminator, self).__init__()
        self.num_layer = num_layer
        self.emb = nn.Embedding(c_dim, c_dim)
        self.input = nn.Linear(x_dim+c_dim, hidden_dim)
        self.inputbn = nn.BatchNorm1d(hidden_dim)
        self.hidden = []
        self.BN = []

        self.wgan = wgan
        for i in range(num_layer):
            fc = nn.Linear(hidden_dim, hidden_dim)
            setattr(self, "fc%d" % i, fc)
            self.hidden.append(fc)
            bn = nn.BatchNorm1d(hidden_dim)
            setattr(self, "bn%d" % i, bn)
            self.BN.append(bn)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, z, c):
        # if self.condition:
        #     assert c is not None
        #     z = torch.cat((z, c), dim=1)
        z = torch.cat([z, self.emb(c)], dim=-1)
        z = self.input(z)
        z = F.leaky_relu(z,inplace=False)
        #z = self.Dropout(z)
        z=F.dropout(z,training=self.training)
        for i in range(len(self.hidden)):
            z = self.hidden[i](z)
            #       z = self.BN[i](z)
          #  z = self.Dropout(z)
            z=F.dropout(z,training=self.training)
            z = F.leaky_relu(z,inplace=False)
        z = self.output(z)
        if self.wgan:
            return z
        else:
            return torch.sigmoid(z)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
