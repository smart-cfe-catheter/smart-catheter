import numpy as np
from torch.utils.data import Dataset, random_split


class CatheterDataset(Dataset):
    def __init__(self, time_series=False, transform=None, target_transform=None):
        self.x = np.empty((0, 3)) if not time_series else np.empty((0, 0, 3))
        self.y = np.empty(0) if not time_series else np.empty((0, 1))
        file_dir = 'data/preprocess'
        file_cnt = 32

        for i in range(file_cnt):
            record = np.loadtxt(f'{file_dir}/{i + 1}.csv', delimiter=',', skiprows=1, usecols=(1, 2, 3, 4))
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


def load_dataset(time_series=False, transform=None, target_transform=None):
    raw_data = CatheterDataset(time_series=time_series, transform=transform, target_transform=target_transform)
    total_cnt = len(raw_data)
    train_cnt = int(total_cnt * 0.8)
    validation_cnt = int(train_cnt * 0.2)

    split_result = random_split(raw_data, [train_cnt, validation_cnt, total_cnt - (train_cnt + validation_cnt)])
    train_data, validation_data, test_data = split_result[0], split_result[1], split_result[2]

    return train_data, validation_data, test_data


if __name__ == "__main__":
    t1, t2, t3 = load_dataset()
    print(t1[:][0].shape, t2[:][0].shape, t3[:][0].shape)
