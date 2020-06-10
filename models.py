import torch
import torchvision
from torch import nn
from torch.nn import functional as f
from torchvision.models import resnet152, resnet101, resnet50, vgg16_bn, vgg19_bn


def stack_linear_layers(dim, layer_cnt):
    return [nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.LeakyReLU()) for _ in range(layer_cnt)]


class DNN(nn.Module):
    def __init__(self, layer_cnt):
        super(DNN, self).__init__()
        self.type = 'DNN'

        layers = stack_linear_layers(3, layer_cnt)
        self.fc_layers = nn.Sequential(*layers)
        self.decoder = nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc_layers(x)
        x = f.leaky_relu(self.decoder(x))
        return x


class RNN(nn.Module):
    def __init__(self, layer_cnt):
        super(RNN, self).__init__()
        self.type = 'RNN'

        self.nlayer = layer_cnt
        self.nhid = 100
        self.rnn = nn.GRU(3, self.nhid, self.nlayer)
        self.decoder = nn.Linear(self.nhid, 1)

    def forward(self, x, h=None):
        x, h = self.rnn(x, h)
        x = f.leaky_relu(self.decoder(x))

        return x, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros((self.nlayer, batch_size, self.nhid))


class CNN(nn.Module):
    def __init__(self, layer_cnt):
        super(CNN, self).__init__()
        self.type = 'CNN'

        self.backbone = resnet50()
        self.decoder = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = f.leaky_relu(self.decoder(x))

        return x
