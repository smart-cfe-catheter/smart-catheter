from os import listdir
from os.path import isfile, join

import numpy as np
from torch.utils.data import Dataset


def import_data(file_name, root_dir):
    x_data = np.loadtxt(f'{root_dir}/input/{file_name}', delimiter=',')
    y_data = np.loadtxt(f'{root_dir}/output/{file_name}', delimiter=',')

    return x_data, y_data


class CatheterDataset(Dataset):
    def __init__(self, folders, type):
        self.type = type

        x = [] if type == 'RNN' else None
        y = [] if type == 'RNN' else None
        max_len = 0
        root_dir = f'data/preprocess/' + ('spectrogram' if type == 'CNN' else 'series')

        for dir_name in folders:
            file_dir = f'{root_dir}/{dir_name}'
            input_dir = file_dir + '/input'
            folder = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and '.csv' in f]

            for file in folder:
                x_data, y_data = import_data(file, file_dir)
                if type == 'CNN':
                    x_data = x_data.reshape((-1, 100, 100, 3))

                if x is None:
                    x, y = x_data, y_data
                elif type == 'RNN':
                    x.append(x_data)
                    y.append(y_data)
                    max_len = max(max_len, x.shape[0])
                else:
                    x = np.append(x, x_data, axis=0)
                    y = np.append(y, y_data, axis=0)
        if type == 'RNN':
            for i in range(len(x)):
                length = x[i].shape[0]
                x[i] = np.pad(x[i], [(max_len - length, 0), (0, 0)], mode='constant', constant_values=0.0)
                y[i] = np.pad(y[i], [(max_len - length, 0), (0, 0)], mode='constant', constant_values=0.0)
            x, y = np.stack(x, axis=0), np.stack(y, axis=0)

        self.x = np.float64(x)
        self.y = np.float64(y)
        self.len = x.shape[1] if type == 'RNN' else x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.type == 'RNN':
            return self.x[:, idx], self.y[:, idx]
        return self.x[idx], self.y[idx]
