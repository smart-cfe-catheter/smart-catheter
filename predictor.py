import argparse

import matplotlib.pyplot as plt
import torch

import dataset
import models


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='BasicNet', choices=['BasicNet', 'FNet', 'RNNNet'])
    parser.add_argument('--file-name', type=str, default='checkpoints/test/checkpoint_final.pth')
    parser.add_argument('--result-dir', type=str, default='results/test')
    args = parser.parse_args()

    time_series = (args.model == 'RNNNet')
    torch.manual_seed(1)
    model = None
    if args.model == 'BasicNet':
        model = models.BasicNet()
    elif args.model == 'FNet':
        model = models.FNet()
    elif args.model == 'RNNNet':
        model = models.RNNNet()
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

        x_data, y_data = torch.from_numpy(x_data), torch.from_numpy(y_data)
        y_pred = model(x_data)
        if time_series:
            y_pred = y_pred[0]

        loss1 = torch.nn.functional.l1_loss(y_data, y_pred)
        loss2 = torch.nn.functional.mse_loss(y_data, y_pred)
        print(f'Prediction {i} L1 loss : {loss1.data.numpy()}N / L2 loss : {loss2.data.numpy()}N')

        y_data, y_pred = y_data.view(-1), y_pred.view(-1)
        x_coors = range(0, y_data.shape[0])
        plt.plot(x_coors, y_data.data, label='real value')
        plt.plot(x_coors, y_pred.data, label='prediction')
        plt.ylabel('Force [N]')
        plt.legend(loc=2)
        plt.savefig(f'{args.result_dir}/prediction-{i}.png', dpi=300)
        plt.cla()


main()
