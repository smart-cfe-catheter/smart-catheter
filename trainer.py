import torch
from torch.nn import functional as f


def repackage_hidden(h):
    if h is None:
        return h
    elif isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class Trainer:
    def __init__(self, model, time_series, rnn, optimizer, device):
        self.model = model
        self.time_series = time_series
        self.rnn = rnn
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, x, y, h=None, reduction='mean'):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()

        if self.rnn:
            h = repackage_hidden(h)
            output, h = self.model(x, h)
        else:
            output = self.model(x)
        loss = f.mse_loss(output, y, reduction=reduction)
        return loss, h

    def train(self, loader, log_interval=100):
        self.model.train()
        total_loss = 0.0

        h = None
        for batch, (x, y) in enumerate(loader):
            if self.time_series:
                x, y = torch.transpose(x, 0, 1), torch.transpose(y, 0, 1)
            loss, h = self.train_epoch(x, y, h=h)
            total_loss += loss.item() * len(x)
            loss.backward()

            self.optimizer.step()
            if batch % log_interval == 0:
                print(f'Batch {batch}/{len(loader)}\t Loss: {loss.item()}')

        return total_loss / len(loader.dataset)

    def test(self, loader):
        self.model.eval()
        total_loss = 0.0

        h = None
        with torch.no_grad():
            for batch, (x, y) in enumerate(loader):
                if self.time_series:
                    x, y = torch.transpose(x, 0, 1), torch.transpose(y, 0, 1)
                loss, h = self.train_epoch(x, y, h=h, reduction='sum')
                total_loss += loss.item()

        return total_loss / len(loader.dataset)
