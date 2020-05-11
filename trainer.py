import torch
from torch.nn import functional as f


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    @staticmethod
    def compute_loss(output, target):
        return f.mse_loss(output, target)

    def train(self, loader, log_interval=100):
        self.model.train()
        total_loss = 0.0

        for batch, (x, y) in enumerate(loader):
            x, y = x.double().to(self.device), y.view(-1, 1).to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)

            loss = self.compute_loss(output, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss

            if batch % log_interval == 0:
                print(f'Batch {batch}/{len(loader)}\t Loss: {loss.item()}')

        return total_loss / len(loader.dataset)

    def test(self, loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.double().to(self.device), y.view(-1, 1).to(self.device)

                output = self.model(x)
                total_loss += self.compute_loss(output, y)

        return total_loss / len(loader.dataset)
