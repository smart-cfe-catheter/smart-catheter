import torch


class ToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x)
