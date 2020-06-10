from torch import nn
from torch.nn import functional as f


def stack_linear_layers(dim, layer_cnt):
    return [nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.LeakyReLU()) for _ in range(layer_cnt)]


class DNN(nn.Module):
    def __init__(self, nlayers):
        super(DNN, self).__init__()
        self.type = 'DNN'

        layers = stack_linear_layers(3, nlayers)
        self.fc_layers = nn.Sequential(*layers)
        self.decoder = nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc_layers(x)
        x = f.leaky_relu(self.decoder(x))
        return x


class RNN(nn.Module):
    def __init__(self, nlayers, nhids):
        super(RNN, self).__init__()
        self.type = 'RNN'

        self.nlayers = nlayers
        self.nhids = nhids
        self.rnn = nn.GRU(3, self.nhids, self.nlayers)
        self.decoder = nn.Linear(self.nhids, 1)

    def forward(self, x, h):
        x, h = self.rnn(x, h)
        x = f.leaky_relu(self.decoder(x))

        return x, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros((self.nlayers, batch_size, self.nhids))


class SigDNN(nn.Module):
    def __init__(self, nlayers):
        super(SigDNN, self).__init__()
        self.type = 'SigDNN'

        layers = stack_linear_layers(300, nlayers)
        self.fc_layers = nn.Sequential(*layers)
        self.decoder = nn.Linear(300, 1)

    def forward(self, x):
        x = self.fc_layers(x)
        x = f.leaky_relu(self.decoder(x))
        return x


class CNN(nn.Module):
    def __init__(self, backbone):
        super(CNN, self).__init__()
        self.type = 'CNN'

        self.backbone = eval(backbone)()
        self.decoder = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = f.leaky_relu(self.decoder(x))

        return x
