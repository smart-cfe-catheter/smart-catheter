from torch import nn
from torch.nn import functional as f


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()

        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        return x
