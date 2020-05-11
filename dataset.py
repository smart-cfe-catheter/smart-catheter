from torch.utils.data import Dataset, random_split
import numpy as np


class CatheterDataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        data_cnt = 60000 if train else 1000
        self.x = np.random.randn(data_cnt, 3)
        self.y = np.random.rand(data_cnt)
        self.len = len(self.x)

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
