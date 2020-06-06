import numpy as np


def noise_cancel(data):
    data -= np.mean(data)
    return data
