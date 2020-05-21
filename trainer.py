import torch
from torch.nn import functional as f


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train(self, loader, log_interval=100):
        self.model.train()
        total_loss = 0.0

        for batch, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(x)
            loss = f.mse_loss(output, y)
            total_loss += loss.item() * len(x)
            loss.backward()

            self.optimizer.step()
            if batch % log_interval == 0:
                print(f'Batch {batch}/{len(loader)}\t Loss: {loss.item()}')

        return total_loss / len(loader.dataset)

    def test(self, loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                output = self.model(x)
                total_loss += f.mse_loss(output, y, reduction='sum').item()

        return total_loss / len(loader.dataset)
