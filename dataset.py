from os import listdir
from os.path import isfile, join

import numpy as np
from scipy.constants import g
from torch.utils.data import Dataset

means = [1539.7412018905536, 1539.6866708934765, 1539.4585310270897]
stds = [0.08834883761864104, 0.118099138737808, 0.09286198197001226]


class CatheterDataset(Dataset):
    def __init__(self, folders, time_series=False, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        x = np.empty((0, 3)) if not time_series else []
        y = np.empty((0, 1)) if not time_series else []
        files = []
        max_len = 0

        for dir_name in folders:
            file_dir = f'data/preprocess/{dir_name}'
            folder = [f'{file_dir}/{f}' for f in listdir(file_dir) if isfile(join(file_dir, f))]
            files += folder

        for file in files:
            record = np.loadtxt(file, delimiter=',', usecols=(0, 1, 2, 3))
            x_data, y_data = record[:, [1, 2, 3]], record[:, [0]]

            for i in range(3):
                x_data[:, i] -= means[i]
                x_data[:, i] /= stds[i]

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
            x, y = np.transpose(x, [1, 0, 2]), np.transpose(y, [1, 0, 2])

        self.x = np.float64(x)
        self.y = y * g * 1e-3
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


def load_dataset(no_validation=False, time_series=False, transform=None, target_transform=None):
    test_set = CatheterDataset(['test'], time_series, transform, target_transform)
    if no_validation:
        train_set = CatheterDataset(['train', 'validation'], time_series, transform, target_transform)
        return train_set, test_set
    else:
        train_set = CatheterDataset(['train'], time_series, transform, target_transform)
        validation_set = CatheterDataset(['validation'], time_series, transform, target_transform)
        return train_set, validation_set, test_set
