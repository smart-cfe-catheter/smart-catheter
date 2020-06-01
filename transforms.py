import torch

means = []
stds = []
medians = []


class ToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x)


class NoiseCancel(object):
    def __call__(self, x):
        return x - torch.mean(x)


class Normalize(object):
    def __call__(self, x):
        for i in range(3):
            x[i] = (x[i] - means[i]) / stds[i]

        return x
