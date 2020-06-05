from os import listdir
from os.path import isfile, join

import numpy as np
import torch
from torch.utils.data import Dataset

import transforms


def import_data(file_name):
    record = np.loadtxt(file_name, delimiter=',', usecols=(0, 1, 2, 3))
    x_data, y_data = record[:, [1, 2, 3]], record[:, [0]]
    x_data = transforms.normalize(x_data)

    return x_data, y_data


class CatheterDataset(Dataset):
    def __init__(self, folders, time_series=False, window=None):
        self.window = window
        self.time_series = time_series
        x = np.empty((0, 3)) if not time_series else []
        y = np.empty((0, 1)) if not time_series else []
        files = []
        max_len = 0

        for dir_name in folders:
            file_dir = f'data/preprocess/{dir_name}'
            folder = [f'{file_dir}/{f}' for f in listdir(file_dir) if isfile(join(file_dir, f))]
            files += folder

        for file in files:
            x_data, y_data = import_data(file)

            if time_series:
                x.append(x_data)
                y.append(y_data)
                max_len = max(x_data.shape[0], max_len)
            else:
                x = np.append(x, x_data, axis=0)
                y = np.append(y, y_data, axis=0)

        if time_series:
            for i in range(len(x)):
                length = x[i].shape[0]
                x[i] = np.pad(x[i], [(0, max_len - length), (0, 0)], mode='constant')
                y[i] = np.pad(y[i], [(0, max_len - length), (0, 0)], mode='constant')
            x, y = np.stack(x, axis=0), np.stack(y, axis=0)

        self.x = torch.from_numpy(np.float64(x))
        self.y = torch.from_numpy(y)
        self.len = self.x.shape[1] if time_series else self.x.shape[0]
        if window is not None:
            self.len = self.len - window + 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.time_series:
            if self.window is None:
                return self.x[:, idx], self.y[:, idx]
            else:
                return self.x[:, idx:idx + self.window], self.y[:, idx + self.window - 1]
        return self.x[idx], self.y[idx]


def load_dataset(no_validation=False, time_series=False, window=None):
    test_set = CatheterDataset(['test'], time_series, window)
    if no_validation:
        train_set = CatheterDataset(['train', 'validation'], time_series, window)
        return train_set, test_set
    else:
        train_set = CatheterDataset(['train'], time_series, window)
        validation_set = CatheterDataset(['validation'], time_series, window)
        return train_set, validation_set, test_set
