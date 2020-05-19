from torch import nn
from torch.nn import functional as f


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()

        self.fc1 = nn.Linear(3, 3)
        self.result_layer = nn.Linear(3, 1)

    def forward(self, x):
        x = f.relu(self.fc1(x))

        return self.result_layer(x)
