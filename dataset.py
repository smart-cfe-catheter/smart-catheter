import numpy as np
from torch.utils.data import Dataset, random_split


class CatheterDataset(Dataset):
    def __init__(self, train=True, time_series=False, transform=None, target_transform=None):
        self.x = np.empty((0, 3))
        self.y = np.empty(0)
        dir_name = 'data/train' if train else 'data/test'
        cnt = 28 if train else 4

        for i in range(cnt):
            record = np.loadtxt(f'{dir_name}/{i + 1}.csv', delimiter=',', skiprows=1, usecols=(1, 2, 3, 4))
            self.x = np.append(self.x, record[:, [1, 2, 3]], axis=0)
            self.y = np.append(self.y, record[:, 0], axis=0)

        self.len = self.x.shape[0]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


def load_dataset(transform=None, target_transform=None):
    raw_data = CatheterDataset(train=True, transform=transform, target_transform=target_transform)
    test_data = CatheterDataset(train=False, transform=transform, target_transform=target_transform)
    total_cnt = len(raw_data)
    train_cnt = int(total_cnt * 0.9)

    split_result = random_split(raw_data, [train_cnt, total_cnt - train_cnt])
    train_data, validation_data = split_result[0], split_result[1]

    return train_data, validation_data, test_data


if __name__ == "__main__":
    data = CatheterDataset()
    print(data[:][0].shape)
