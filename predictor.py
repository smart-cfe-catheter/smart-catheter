import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as f
from scipy.interpolate import interp1d

from models import RNN


class Predictor:
    """
    model : model file for prediction. (e.g. checkpoints/test/model.pth)
    device : if you want to use gpu, set to 'gpu'
    """
    def __init__(self, model, device='cpu'):
        self.device = device
        self.model = RNN(4, 100).to(device)
        self.h = self.model.init_hidden(1).to(device)
        
        state_dict = torch.load(model, map_location='cpu')
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.model.eval()

    """
    x : numpy array shape of (N, 3).
    t : time array shape of (N). Must be strictly ascending order
    """
    def predict(self, x, t):
        with torch.no_grad():
            t -= t[0]
            interpolator = interp1d(t, x, axis=0)
            x = interpolator(np.arange(0, 0.1, 0.001))
            x = x.reshape((-1, 1, 3))

            x = torch.from_numpy(np.float32(x)).to(self.device)

            output, self.h = self.model(x, self.h)
            return output.data.reshape(-1)
