import argparse

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn import init
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
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-model', action='store_true', default=False)
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--model', type=str, default='BasicNet', choices=['BasicNet', 'FNet'])
parser.add_argument('--device-ids', type=int, nargs='+', default=None)
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoint')
parser.add_argument('--save-per-epoch', type=int, default=5)
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device('cuda' if use_cuda else 'cpu')
print(f'device selected: {device}\n')

train_data, validation_data, test_data = load_dataset(transform=tf.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size)
validation_loader = DataLoader(dataset=validation_data, batch_size=args.batch_size)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size)

model = models.BasicNet()
if args.model == 'FNet':
    model = models.FNet()

print(f'Selected {model}')
model = model.to(device).double().apply(weight_init)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
trainer = Trainer(model, optimizer=optimizer, device=device)

train_losses = []
validation_losses = []
for e in range(args.epochs):
    print(f'<Train Epoch #{e + 1}>')
    train_loss = trainer.train(train_loader, log_interval=args.log_interval)
    validation_loss = trainer.test(validation_loader)

    train_losses.append(train_loss)
    validation_losses.append(validation_loss)
    scheduler.step(validation_loss)
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

