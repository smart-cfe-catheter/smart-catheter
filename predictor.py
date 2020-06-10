import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as f
from scipy.interpolate import interp1d


class SigRNN(nn.Module):
    def __init__(self, nlayers, nhids):
        super(SigRNN, self).__init__()
        self.type = 'SigRNN'

        self.nlayers = nlayers
        self.nhids = nhids
        self.rnn = nn.GRU(3 * 100 * 100, self.nhids, self.nlayers)
        self.decoder = nn.Linear(self.nhids, 1)

    def forward(self, x, h):
        x, h = self.rnn(x, h)
        x = f.leaky_relu(self.decoder(x))

        return x, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros((self.nlayers, batch_size, self.nhids))


class Predictor:
    """
    model : model file for prediction. (e.g. checkpoints/test/model.pth)
    device : if you want to use gpu, set to 'gpu'
    """
    def __init__(self, model, device='cpu'):
        self.device = device
        self.model = SigRNN(1, 100).to(device)
        self.h = self.model.init_hidden(1)
        
        state_dict = torch.load(self.model, map_location='cpu')
        self.model.load_state_dict(state_dict['model_state_dict'])

    """
    x : numpy array shape of (N, 3).
    t : time array shape of (N). Must be strictly ascending order
    """
    def predict(self, x, t):
        t -= t[0]
        interpolator = interp1d(t, x, axis=0)
        x = interpolator(np.arange(0, 0.1, 0.001))
        input = np.empty((99, 100, 3))

        for channel in range(3):
            result, _ = pywt.cwt(x[:, channel], range(1, 100), 'morl', 0.001)
            input[:, :, channel] = result
        input = np.append(input, x.reshape(1, 100, 3), axis=0).reshape(1, 1, 300)

        output, self.h = self.model(input, self.h)
        
        return output.data
