import torch
from torch.nn import functional as f


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class Trainer:
    def __init__(self, model, optimizer=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, x, y, h=None, reduction='mean'):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()

        if self.model.type == 'RNN':
            h = repackage_hidden(h).to(self.device)
            output, h = self.model(x, h)
            loss = f.smooth_l1_loss(output, y, reduction=reduction)
            return loss, h
        else:
            output = self.model(x)
            loss = f.smooth_l1_loss(output, y, reduction=reduction)
            return loss

    def train(self, loader, log_interval=100):
        self.model.train()
        total_loss = 0.0
        total_num = 0

        for batch, (x, y) in enumerate(loader):
            if self.model.type == 'RNN':
                if batch == 0:
                    h = self.model.init_hidden(x.shape[1])
                loss, h = self.train_epoch(x, y, h=h, reduction='mean')
            else:
                loss = self.train_epoch(x, y, reduction='mean')
            total_loss += loss.item() * y.numel()
            loss.backward()
            total_num += y.numel()

            self.optimizer.step()
            if batch % log_interval == 0:
                print(f'Batch {batch}/{len(loader)}\t Loss: {total_loss / total_num}')

        return total_loss / total_num

    def test(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_num = 0

        with torch.no_grad():
            for batch, (x, y) in enumerate(loader):
                if self.model.type == 'RNN':
                    if batch == 0:
                        h = self.model.init_hidden(x.shape[1])
                    loss, h = self.train_epoch(x, y, h=h, reduction='sum')
                else:
                    loss = self.train_epoch(x, y, reduction='sum')
                total_loss += loss.item()
                total_num += y.numel()

        return total_loss / total_num
