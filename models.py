import torch
from torch import nn
from torch.nn import functional as f


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()

        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 3)
        self.result_layer = nn.Linear(3, 1)

    def forward(self, x):
        x = f.leaky_relu(self.fc1(x))
        x = f.leaky_relu(self.fc2(x))
        return self.result_layer(x)

    def __str__(self):
        return 'BasicNet'


class FNet(nn.Module):
    def __init__(self):
        super(FNet, self).__init__()

        self.degree = 5
        self.Fh = nn.Linear(3, 2)

        self.fc1 = nn.Linear(self.degree + 1, self.degree + 1)
        self.fc2 = nn.Linear(self.degree + 1, self.degree + 1)
        
        self.result_layer = nn.Linear(self.degree + 1, 1)

    def forward(self, x):
        x = self.Fh(x)
        x = torch.norm(x, dim=1)

        x = x.unsqueeze(1)
        x = torch.cat([x ** i for i in range(0, self.degree + 1)], 1)

        x = f.leaky_relu(self.fc1(x))
        x = f.leaky_relu(self.fc2(x))

        return self.result_layer(x)

    def __str__(self):
        return 'FNet'

