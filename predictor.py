import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import g

from models import BasicNet, FNet


def import_data(num):
    record = np.loadtxt(f'data/preprocess/{num}.csv', delimiter=',', usecols=(0, 1, 2, 3))
    return torch.from_numpy(record[:, [1, 2, 3]]), torch.from_numpy(record[:, 0] * g * 1e-3)


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='BasicNet', choices=['BasicNet', 'FNet'])
    parser.add_argument('--file-name', type=str, default='checkpoints/test/checkpoint_final.pth')
    parser.add_argument('--result-dir', type=str, default='results')
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

    for i in range(1, 33):
        x_data, y_data = import_data(i)
        y_pred = model(x_data).view(-1)
        loss1 = torch.nn.functional.l1_loss(y_data, y_pred)
        loss2 = torch.nn.functional.mse_loss(y_data, y_pred)
        print(f'Prediction {i} L1 loss : {loss1.data.numpy()}, L2 loss : {loss2.data.numpy()}')

        x_coors = range(0, y_data.shape[0])
        plt.plot(x_coors, y_data.data, label='real value')
        plt.plot(x_coors, y_pred.data, label='prediction')
        plt.ylabel('Force [N]')
        plt.legend(loc=2)
        plt.savefig(f'{args.result_dir}/prediction-{i}.png', dpi=300)
        plt.cla()


main()
