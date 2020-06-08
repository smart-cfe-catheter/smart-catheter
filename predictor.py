import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import models
from data import import_data


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='DNN', choices=['DNN', 'SigDNN'])
    parser.add_argument('--file-name', type=str, default='checkpoints/test/checkpoint_final.pth')
    parser.add_argument('--result-dir', type=str, default='results/test')
    parser.add_argument('--layer-cnt', type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(1)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    model = {
        'DNN': models.DNN(args.layer_cnt),
        'RNN': models.RNN(args.layer_cnt),
        'CNN': models.CNN(args.layer_cnt)
    }.get(args.model, 'DNN')
    model = model.double()
    loaded_state_dict = torch.load(args.file_name, map_location='cpu')
    try:
        model.load_state_dict(loaded_state_dict['model_state_dict'])
    except RuntimeError:
        model.module.load_state_dict(loaded_state_dict['model_state_dict'])
    model.eval()

    with torch.no_grad():
        for idx in range(1, 13):
            root_dir = 'data/preprocess/spectrogram/test' if model.type == 'CNN' else 'data/preprocess/series/test'
            x, y = import_data(f'{root_dir}/{idx}.csv', model.settings)
            if model.type == 'CNN':
                x = x.reshape((-1, 100, 100, 3))
            elif model.type == 'RNN':
                x = x.reshape((1, x.shape[0], 3))

            x, y = torch.from_numpy(x), torch.from_numpy(y)

            y_pred = model(x)
            y, y_pred = y.view(-1), y_pred.view(-1)
            loss = torch.nn.functional.l1_loss(y, y_pred)
            print(f'Record #{idx} L1 Loss : {loss}')

            xrange = range(0, y.shape[0])
            plt.plot(xrange, y.data, label='real value')
            plt.plot(xrange, y_pred.data, label='prediction')
            plt.ylabel('Weight [g]')
            plt.legend(loc=2)
            plt.savefig(f'{args.result_dir}/prediction-{idx}.png', dpi=300)
            plt.cla()


main()
