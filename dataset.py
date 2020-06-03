import numpy as np
from scipy.constants import g
from torch.utils.data import Dataset, random_split


class CatheterDataset(Dataset):
    def __init__(self, time_series=False, transform=None, target_transform=None):
        self.time_series = time_series
        self.transform = transform
        self.target_transform = target_transform

        x = np.empty((0, 3)) if not time_series else []
        y = np.empty((0, 1)) if not time_series else []
        file_dir = 'data/preprocess'
        file_cnt = 84
        max_len = 0

        for i in range(file_cnt):
            record = np.loadtxt(f'{file_dir}/{i + 1}.csv', delimiter=',', usecols=(0, 1, 2, 3, 4))
            y_data = record[:, [0]]

            if time_series:
                x_data = record[:, [1, 2, 3, 4]]
                x.append(x_data)
                y.append(y_data)
                max_len = max(x_data.shape[0], max_len)
            else:
                x_data = record[:, [1, 2, 3]]
                x = np.append(x, x_data, axis=0)
                y = np.append(y, y_data, axis=0)

        if time_series:
            for i in range(file_cnt):
                length = x[i].shape[0]
                x[i] = np.pad(x[i], [(0, max_len - length), (0, 0)], mode='constant')
                y[i] = np.pad(y[i], [(0, max_len - length), (0, 0)], mode='constant')
            x, y = np.stack(x, axis=0), np.stack(y, axis=0)

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


class RNNCatheterDataset(Dataset):
    def __init__(self, data, transform, target_transform):
        self.transform = transform
        self.target_transform = target_transform

        self.x = np.transpose(data[0], [1, 0, 2])
        self.y = np.transpose(data[1], [1, 0, 2])
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
    raw_data = CatheterDataset(time_series=time_series, transform=transform, target_transform=target_transform)
    _ = raw_data[:]

    total_cnt = len(raw_data)
    train_cnt = int(total_cnt * 0.8)
    if no_validation:
        split_result = random_split(raw_data, [train_cnt, total_cnt - train_cnt])
        if time_series:
            return RNNCatheterDataset(split_result[0][:], transform, target_transform), \
                   RNNCatheterDataset(split_result[1][:], transform, target_transform)
        return split_result[0], split_result[1]
    else:
        validation_cnt = int(train_cnt * 0.2)

        split_result = random_split(raw_data, [train_cnt, validation_cnt, total_cnt - (train_cnt + validation_cnt)])
        if time_series:
            return RNNCatheterDataset(split_result[0][:], transform, target_transform), \
                   RNNCatheterDataset(split_result[1][:], transform, target_transform), \
                   RNNCatheterDataset(split_result[2][:], transform, target_transform)
        return split_result[0], split_result[1], split_result[2]
