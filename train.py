import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torch.nn import init
from torch.utils.data import DataLoader

import models
from dataset import load_dataset
from trainer import Trainer


def weight_init(model):
    name = model.__class__.__name__
    if name.find('Linear') != -1:
        init.kaiming_normal_(model.weight.data)


def save_checkpoint(tag, args, epoch, model, optimizer, scheduler):
    print('Snapshot Checkpoint...')
    if args.device_ids and len(args.device_ids) > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'last_epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, os.path.join(args.checkpoint_dir, 'checkpoint_' + str(tag) + '.pth'))


def load_checkpoint(args, model, optimizer, scheduler):
    last_epoch = 0
    if os.path.isdir(args.checkpoint_dir) is False:
        os.makedirs(args.checkpoint_dir)
        return last_epoch

    files = os.listdir(args.checkpoint_dir)
    files.sort(key=len)

    if len(files) == 0:
        return last_epoch

    load_dir = os.path.join(args.checkpoint_dir, files[-1])
    if os.path.isfile(load_dir) is True:
        loaded_state_dict = torch.load(load_dir, map_location='cpu')
        print("Checkpoint loaded from " + load_dir)

        last_epoch = loaded_state_dict['last_epoch']
        optimizer.load_state_dict(loaded_state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(loaded_state_dict['scheduler_state_dict'])

        try:
            model.load_state_dict(loaded_state_dict['model_state_dict'])
        except RuntimeError:
            model.module.load_state_dict(loaded_state_dict['model_state_dict'])

    else:
        print("Checkpoint load failed")

    return last_epoch


def main():
    parser = argparse.ArgumentParser(description='Smart Catheter Trainer')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='BasicNet', choices=['BasicNet', 'FNet', 'RNNNet'])
    parser.add_argument('--device-ids', type=int, nargs='+', default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/test')
    parser.add_argument('--save-per-epoch', type=int, default=5)
    parser.add_argument('--noise-cancel', action='store_true', default=False)
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    time_series = (args.model == 'RNNNet')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'device selected: {device}\n')

    train_data, validation_data, test_data = load_dataset(time_series=time_series)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=not time_series)
    validation_loader = DataLoader(dataset=validation_data, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size)

    model = None
    if args.model == 'BasicNet':
        model = models.BasicNet()
    elif args.model == 'FNet':
        model = models.FNet()
    elif args.model == 'RNNNet':
        model = models.RNNNet()
    if args.device_ids and use_cuda and len(args.device_ids) > 1:
        model = nn.DataParallel(model, device_ids=[i for i in range(len(args.device_ids))])
    model = model.to(device).double().apply(weight_init)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    last_epoch = load_checkpoint(args, model, optimizer, scheduler) if not args.reset else 0

    trainer = Trainer(model, time_series=time_series, optimizer=optimizer, device=device)

    train_losses = []
    validation_losses = []
    for e in range(last_epoch + 1, args.epochs + 1):
        print(f'<Train Epoch #{e}>')
        train_loss = trainer.train(train_loader, log_interval=args.log_interval)
        validation_loss = trainer.test(validation_loader)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        scheduler.step(validation_loss)

        if args.save_model and e % args.save_per_epoch == 0:
            save_checkpoint(e, args, e, model, optimizer, scheduler)
        print(f'Train Loss: {train_loss}N / Validation Loss: {validation_loss}N\n')
    print(f'\nTest Loss: {trainer.test(test_loader)}N')

    plt.plot(range(last_epoch + 1, args.epochs + 1), train_losses, label='train loss')
    plt.plot(range(last_epoch + 1, args.epochs + 1), validation_losses, label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss [N]')
    plt.legend(loc=2)

    if args.save_model:
        save_checkpoint('final', args, args.epochs, model, optimizer, scheduler)
        plt.savefig(f'{args.checkpoint_dir}/learning-curve.png', dpi=300)

    if args.visualize:
        plt.show()


main()
