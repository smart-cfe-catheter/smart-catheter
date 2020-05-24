import time
import argparse
import matplotlib.pyplot as plt
import torch
from torch.nn import init
from torch.nn import functional as f
from torch.utils.data import DataLoader

import models
import transforms as tf
from dataset import load_dataset
from trainer import Trainer


def weight_init(model):
    name = model.__class__.__name__
    if name.find('Conv') != -1 or name.find('Linear') != -1:
        init.xavier_normal_(model.weight.data)


parser = argparse.ArgumentParser(description='Smart Catheter Trainer')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default: 1e-6)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--visualize', action='store_true', default=False,
                    help='For Showing the learning curve')
parser.add_argument('--model-name', type=str, default='basicnet', metavar='N',
                    help='Model going to be trained. There are basicnet and lenet')
parser.add_argument('--file-name', type=str, default='model', metavar='N',
                    help='File name where the model weight will be saved.')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')
print(f'device selected: {device}\n')

model = models.BasicNet().to(device).double().apply(weight_init)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
trainer = Trainer(model, optimizer=optimizer, device=device)

train_data, validation_data, test_data = load_dataset(transform=tf.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size)
validation_loader = DataLoader(dataset=validation_data, batch_size=args.batch_size)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size)

train_losses = []
validation_losses = []
for e in range(args.epochs):
    print(f'<Train Epoch #{e + 1}>')
    train_loss = trainer.train(train_loader, log_interval=args.log_interval)
    validation_loss = trainer.test(validation_loader)

    train_losses.append(train_loss)
    validation_losses.append(validation_loss)
    print(f'Train Loss: {train_loss} / Validation Loss: {validation_loss}\n')

print(f'\n<Final Losses>\n'
      f'- train: {trainer.test(train_loader)}\n'
      f'- validation: {trainer.test(validation_loader)}\n'
      f'- test: {trainer.test(test_loader)}')

plt.plot(range(1, args.epochs + 1), train_losses, label='train loss')
plt.plot(range(1, args.epochs + 1), validation_losses, label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc=2)
if args.visualize:
    plt.show()

if args.save_model:
    torch.save(model.state_dict(), f'models/{args.file_name}.pt')
    plt.savefig(f'figures/{args.file_name}.png', dpi=300)

