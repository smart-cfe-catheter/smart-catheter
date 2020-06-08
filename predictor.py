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
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='DNN', choices=['DNN', 'RNN', 'CNN'])
    parser.add_argument('--file-name', type=str, default='checkpoints/test/checkpoint_final.pth')
    parser.add_argument('--result-dir', type=str, default='results/test')
    parser.add_argument('--layer-cnt', type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(1)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    model = eval(args.model)(args.layer_cnt)
    loaded_state_dict = torch.load(args.file_name, map_location='cpu')
    try:
        model.load_state_dict(loaded_state_dict['model_state_dict'])
    except RuntimeError:
        model.module.load_state_dict(loaded_state_dict['model_state_dict'])
    model.eval()

    with torch.no_grad():
        root_dir = 'data/preprocess/' + ('spectrogram' if model.type == 'CNN' else 'series')
        x, y = import_data(root_dir, 'test', model.type)
        if model.type == 'RNN':
            x, y = x.transpose((1, 0, 2)), y.transpose((1, 0, 2))
        
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        print(x.shape, y.shape)

        if model.type == 'RNN':
            h = model.init_hidden(ndata['test'])
            h = repackage_hidden(h)
            y_pred, _ = model(x, h)
        else:
            y_pred = model(x)
        
        
        sz = int(y.shape[0] / ndata['test'])
        for idx in range(ndata['test']):
            if model.type == 'RNN':
                real, pred = y[:, idx], y_pred[:, idx]
            else:
                real, pred = y[idx*sz:(idx+1)*sz], y_pred[idx*sz:(idx+1)*sz]
            real, pred = real.view(-1), pred.view(-1)
            loss = l1_loss(real, pred)
            print(f'Record #{idx} L1 Loss : {loss}')

            xrange = range(0, real.shape[0])
            plt.plot(xrange, real.data, label='real value')
            plt.plot(xrange, pred.data, label='prediction')
            plt.ylabel('Weight [g]')
            plt.legend(loc=2)
            plt.savefig(f'{args.result_dir}/prediction-{idx + 1}.png', dpi=300)
            plt.cla()
        print(f'Total Test L1 Loss : {l1_loss(y, y_pred).item()}')


main()
