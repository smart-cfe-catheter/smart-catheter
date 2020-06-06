import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

import dataset
import models


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='DNN', choices=['DNN', 'HFDNN', 'RNN', 'SigDNN'])
    parser.add_argument('--file-name', type=str, default='checkpoints/test/checkpoint_final.pth')
    parser.add_argument('--result-dir', type=str, default='results/test')
    parser.add_argument('--layer-cnt', type=int, default=2)
    args = parser.parse_args()

    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    time_series = (args.model == 'RNN' or args.model == 'SigDNN')
    torch.manual_seed(1)
    model = {'DNN': models.DNN(args.layer_cnt),
             'HFDNN': models.HFDNN(args.layer_cnt),
             'RNN': models.RNN(args.layer_cnt),
             'SigDNN': models.SigDNN(args.layer_cnt)}.get(args.model, 'DNN')
    model = model.double()
    loaded_state_dict = torch.load(args.file_name, map_location='cpu')
    try:
        model.load_state_dict(loaded_state_dict['model_state_dict'])
    except RuntimeError:
        model.module.load_state_dict(loaded_state_dict['model_state_dict'])
    model.eval()

    for i in range(1, 8):
        x_data, y_data = dataset.import_data(f'data/preprocess/test/{i}.csv')
        if time_series:
            x_data, y_data = x_data.reshape(1, -1, 3), y_data.reshape(1, -1, 1)
            if args.model == 'SigDNN':
                sig_len = x_data.shape[1]
                new_x_data, new_y_data = np.empty((0, 1, 10, 3)), np.empty((0, 1, 1))
                for idx in range(sig_len - 9):
                    x, y = x_data[:, idx:idx + 10], y_data[:, idx + 9]
                    x, y = x.reshape(1, 1, 10, 3), y.reshape(1, 1, 1)
                    new_x_data = np.append(new_x_data, x, axis=0)
                    new_y_data = np.append(new_y_data, y, axis=0)
                x_data, y_data = np.transpose(new_x_data, [1, 0, 2, 3]), np.transpose(new_y_data, [1, 0, 2])
                x_data, y_data = x_data.reshape(-1, 30), y_data.reshape(-1, 1)

        x_data, y_data = torch.from_numpy(x_data), torch.from_numpy(y_data)
        y_pred = model(x_data)
        if args.model == 'RNN':
            y_pred = y_pred[0]

        loss = torch.nn.functional.l1_loss(y_data, y_pred)
        print(f'Prediction {i} L1 loss : {loss.data.numpy()}')

        y_data, y_pred = y_data.view(-1), y_pred.view(-1)
        x_coors = range(0, y_data.shape[0])
        plt.plot(x_coors, y_data.data, label='real value')
        plt.plot(x_coors, y_pred.data, label='prediction')
        plt.ylabel('Weight [g]')
        plt.legend(loc=2)
        plt.savefig(f'{args.result_dir}/prediction-{i}.png', dpi=300)
        plt.cla()


main()
