import torch
from torch.nn import functional as f


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class Trainer:
    def __init__(self, model, time_series, optimizer, device):
        self.model = model
        self.time_series = time_series
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, x, y, h=None, reduction='mean'):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()

        if self.time_series:
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
            if self.time_series and batch == 0:
                h = torch.zeros(5, x.shape[1], 3).to(self.device).double()
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
                if self.time_series and batch == 0:
                    h = torch.zeros(5, x.shape[1], 3).to(self.device).double()
                loss, h = self.train_epoch(x, y, h=h, reduction='sum')
                total_loss += loss.item()

        return total_loss / len(loader.dataset)
