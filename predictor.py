import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn.functional import l1_loss

from models import DNN, RNN, CNN
from data import import_data
from preprocess import ndata, frequency


def repackage_hidden(h):
    if h is None:
        return h
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def main():
    parser = argparse.ArgumentParser(description='Smart Catheter Predictor')
    parser.add_argument('--file-name', type=str, default='checkpoints/test/checkpoint_final.pth')
    parser.add_argument('--result-dir', type=str, default='results/test')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(1)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device selected: {device}\n')

    model = torch.load(args.file_name, map_location='cpu').to(device)
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        root_dir = 'data/preprocess/' + ('spectrogram' if model.type == 'CNN' else 'series')
        x, y = import_data(root_dir, 'test', model.type)
        if model.type == 'RNN':
            x, y = x.transpose((1, 0, 2)), y.transpose((1, 0, 2))
        sz = int(y.shape[0] / ndata['test'])

        for idx in range(ndata['test']):
            if model.type == 'RNN':
                x_data, y_real = x[idx], y[idx]
            else:
                x_data, y_real = x[idx*sz:(idx + 1)*sz], y[idx*sz:(idx + 1)*sz]
            x_data, y_real = torch.from_numpy(x_data).to(device), torch.from_numpy(y_real).to(device)

            if model.type == 'RNN':
                h = model.init_hidden(ndata['test']).to(device)
                h = repackage_hidden(h)
                y_pred, _ = model(x_data, h)
            else:
                y_pred = model(x_data)

            y_real, y_pred = y_real.cpu(), y_pred.cpu()
            xrange = range(0, y_real.shape[0])

            loss = l1_loss(y_real, y_pred)
            total_loss += loss.item()
            print(f'Record #{idx + 1} L1 Loss : {loss}')
            
            plt.plot(xrange, y_real.data, label='real value')
            plt.plot(xrange, y_pred.data, label='prediction')
            plt.ylabel('Weight [g]')
            plt.legend(loc=2)
            plt.savefig(f'{args.result_dir}/prediction-{idx + 1}.png', dpi=300)
            plt.cla()
        
        print(f'Total Test L1 Loss : {total_loss / ndata["test"]}')


main()
