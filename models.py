from torch import nn
from torch.nn import functional as f
from torchvision.models import vgg19_bn, resnet18, resnet50, resnet101, resnet152

from preprocess import frequency


def stack_linear_layers(dim, layer_cnt):
    return [nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.LeakyReLU(), nn.Dropout(0.5)) for _ in range(layer_cnt)]


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
        self.rnn.flatten_parameters()
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

        layers = stack_linear_layers(3 * frequency, nlayers)
        self.fc_layers = nn.Sequential(*layers)
        self.decoder = nn.Linear(3 * frequency, 1)

    def forward(self, x):
        x = self.fc_layers(x)
        x = f.leaky_relu(self.decoder(x))
        return x


class SigRNN(nn.Module):
    def __init__(self, nlayers, nhids):
        super(SigRNN, self).__init__()
        self.type = 'SigRNN'

        self.nlayers = nlayers
        self.nhids = nhids
        self.rnn = nn.GRU(3 * frequency * frequency, self.nhids, self.nlayers)
        self.decoder = nn.Linear(self.nhids, 1)

    def forward(self, x, h):
        x, h = self.rnn(x, h)
        x = f.leaky_relu(self.decoder(x))

        return x, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros((self.nlayers, batch_size, self.nhids))


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
