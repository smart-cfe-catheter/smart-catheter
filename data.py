from os import listdir
from os.path import isfile, join

import numpy as np
from torch.utils.data import Dataset

from preprocess import frequency, ndata


def import_data(prefix, split, type):
    fx = open(f'{prefix}/{split}_signal.in', 'rb')
    fy = open(f'{prefix}/{split}_scale.in', 'rb')

    x_buffer, y_buffer = bytearray(fx.read()), bytearray(fy.read())
    x_data, y_data = np.frombuffer(x_buffer, np.float32), np.frombuffer(y_buffer, np.float32)
    if type == 'DNN':
        x_data, y_data = x_data.reshape((-1, 3)), y_data.reshape((-1, 1))
    elif type == 'RNN':
        x_data, y_data = x_data.reshape((ndata[split], -1, 3)), y_data.reshape((ndata[split], -1, 1))
    elif type == 'CNN':
        x_data, y_data = x_data.reshape((-1, frequency, frequency, 3)), y_data.reshape((-1, 1))
        x_data = x_data.transpose((0, 3, 1, 2))
    return x_data, y_data


class CatheterDataset(Dataset):
    def __init__(self, splits, type):
        self.type = type

        x, y = None, None
        if type == 'CNN':
            base_dir = 'data/preprocess/spectrogram'
        else:
            base_dir = 'data/preprocess/series'
        for split_name in splits:
            x_data, y_data = import_data(base_dir, split_name, type)
            if x is None:
                x, y = x_data, y_data
            else:
                x, y = np.append(x, x_data, axis=0), np.append(y, y_data, axis=0)
        
        self.x = x
        self.y = y
        self.len = self.x.shape[1] if type == 'RNN' else self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.type == 'RNN':
            return self.x[:, idx], self.y[:, idx]
        return self.x[idx], self.y[idx]
