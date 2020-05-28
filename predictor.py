import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import g

from models import BasicNet, FNet


def import_data(num):
    record = np.loadtxt(f'data/preprocess/{num}.csv', delimiter=',', skiprows=1, usecols=(1, 2, 3, 4))
    return torch.from_numpy(record[:, [1, 2, 3]]), torch.from_numpy(record[:, 0] * g * 1e-3)


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', type=str, default='BasicNet', choices=['BasicNet', 'FNet'])
parser.add_argument('--file-name', type=str, default='checkpoints/test/checkpoint_final.pth')
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--predict', type=int, default=1)
args = parser.parse_args()

torch.manual_seed(1)
model = BasicNet() if args.model == 'BasicNet' else FNet()
model = model.double()
loaded_state_dict = torch.load(args.file_name, map_location='cpu')
try:
    model.load_state_dict(loaded_state_dict['model_state_dict'])
except RuntimeError:
    model.module.load_state_dict(loaded_state_dict['model_state_dict'])
model.eval()

for i in range(1, args.predict + 1):
    plt.figure(i)
    x_data, y_data = import_data(i)
    y_pred = model(x_data).view(-1)
    loss = torch.nn.functional.mse_loss(y_data, y_pred)
    print(loss.data.numpy())

    x_coors = range(0, y_data.shape[0])
    plt.plot(x_coors, y_data.data, label='real value')
    plt.plot(x_coors, y_pred.data, label='prediction')
    plt.ylabel('Force [N]')
    plt.legend(loc=2)

if args.visualize:
    plt.show()
