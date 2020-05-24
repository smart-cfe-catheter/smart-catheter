import torch


class ToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x)


class Normalize(object):
    def __call__(self, x):
        x[0] = (x[0] - 0.004853980125330318) / 0.010268803725935298
        x[1] = (x[1] - 0.0019010093676787995) / 0.13306334656559207
        x[2] = (x[2] - 0.0021694139502547744) / 0.031074168314918445

        return x
