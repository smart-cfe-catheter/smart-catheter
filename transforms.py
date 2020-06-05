import torch

means = [1539.7412018905536, 1539.6866708934765, 1539.4585310270897]
stds = [0.08834883761864104, 0.118099138737808, 0.09286198197001226]


def normalize(data):
    for i in range(3):
        data[:, i] -= means[i]
        data[:, i] /= stds[i]
    return data


def noise_cancel(data):
    data -= torch.mean(data)
    return data
