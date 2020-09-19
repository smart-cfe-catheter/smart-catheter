import itertools
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
from torch.utils.data import Dataset


class CatheterDataset(Dataset):
    def __init__(self, data_path):
        self.signals = np.load(f"{data_path}/signals.npz")["arr_0"]
        self.scales = np.load(f"{data_path}/scales.npz")["arr_0"]

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        return self.signals[idx], self.scales[idx]

class FilterDataset(CatheterDataset):
    def __init__(self, data_path, cutoff, freq=100, ftype='high'):
        super().__init__(data_path)
        
        nyq = 0.5 * freq
        numer, denom = butter(10, cutoff/nyq, btype=ftype, analog=False)
        self.signals = filtfilt(numer, denom, self.signals.reshape(1,-1)) 


class RNNDataset(CatheterDataset):
    def __init__(self, data_path):
        super().__init__(data_path)

        self.signals = self.signals.reshape((self.signals.shape[0], -1, 300))


class FCNDataset(CatheterDataset):
    def __init__(self, data_path):
        super().__init__(data_path)

        self.signals = self.signals.reshape((-1, 300))
        self.scales = self.scales.reshape((-1, 1))
