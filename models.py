import torch
from torch import nn
from torch.nn import functional as f


def stack_linear_layers(dim, layer_cnt):
    return [nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.LeakyReLU()) for _ in range(layer_cnt)]


class DNN(nn.Module):
    def __init__(self, layer_cnt):
        super(DNN, self).__init__()

        layers = stack_linear_layers(3, layer_cnt)
        self.fc_layers = nn.Sequential(*layers)
        self.decoder = nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc_layers(x)
        x = f.leaky_relu(self.decoder(x))
        return x


class HFDNN(nn.Module):
    def __init__(self, layer_cnt):
        super(HFDNN, self).__init__()

        self.degree = 5
        self.Fh = nn.Linear(3, 2, bias=False)

        layers = stack_linear_layers(self.degree + 1, layer_cnt)
        self.fc_layers = nn.Sequential(*layers)
        self.decoder = nn.Linear(self.degree + 1, 1)

    def forward(self, x):
        x = self.Fh(x)
        x = torch.norm(x, dim=1)

        x = x.unsqueeze(1)
        x = torch.cat([x ** i for i in range(0, self.degree + 1)], 1)

        x = self.fc_layers(x)
        x = f.leaky_relu(self.decoder(x))

        return x


class RNN(nn.Module):
    def __init__(self, layer_cnt):
        super(RNN, self).__init__()

        self.num_layers = layer_cnt
        self.rnn = nn.GRU(3, 3, self.num_layers, batch_first=True)
        self.decoder = nn.Linear(3, 1)

    def forward(self, x, h=None):
        x, h = self.rnn(x, h)
        x = self.decoder(x)

        return x, h

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, 3)


class SigDNN(nn.Module):
    def __init__(self, layer_cnt):
        super(SigDNN, self).__init__()

        layers = stack_linear_layers(30, layer_cnt)
        self.fc_layers = nn.Sequential(*layers)
        self.decoder = nn.Linear(30, 1)

    def forward(self, x):
        x = self.fc_layers(x)
        x = f.leaky_relu(self.decoder(x))
        return x
