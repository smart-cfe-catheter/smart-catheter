import numpy as np
import pywt
import torch
from scipy.interpolate import interp1d


class Predictor:
    """
    model : model file for prediction. (e.g. checkpoints/test/model.pth)
    device : if you want to use gpu, set to 'gpu'
    """

    def __init__(self, model, device='cpu'):
        self.device = device
        self.model = torch.load(model, map_location='cpu').to(device)

    """
    x : numpy array shape of (N, 3).
    t : time array shape of (N). Must be strictly ascending order
    """

    def predict(self, x, t):
        t -= t[0]
        interpolator = interp1d(t, x, axis=0)
        x = interpolator(np.arange(0, 0.1, 0.001))
        input = np.empty((100, 100, 3))

        for channel in range(3):
            result, _ = pywt.cwt(x[:, channel], range(1, 101), 'morl', 0.001)
            input[:, :, channel] = result

        input = torch.from_numpy(input.transpose((2, 0, 1))).to(self.device)
        print(input.shape)
        return self.model(input).data
