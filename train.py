import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torch.nn import init
from torch.utils.data import DataLoader

from data import CatheterDataset
from models import RNN, CNN
from trainer import Trainer


def weight_init(model):
    name = model.__class__.__name__
    if name.find('Linear') != -1 or name.find('Conv') != -1:
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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='DNN', choices=['DNN', 'RNN', 'SigDNN', 'CNN'])
    parser.add_argument('--device-ids', type=int, nargs='+', default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/test')
    parser.add_argument('--save-per-epoch', type=int, default=50)
    parser.add_argument('--reset', action='store_true', default=False)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--nhids', type=int, default=100)
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet101', 'resnet152', 'vgg19_bn'])
    args = parser.parse_args()

    torch.manual_seed(1)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if args.model == 'DNN' or args.model == 'SigDNN':
        model = eval(args.model)(args.nlayers)
    elif args.model == 'RNN':
        model = RNN(args.nlayers, args.nhids)
    else:
        model = CNN(args.backbone)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device selected: {device}\n')

    train_data = CatheterDataset(['train'], model.type)
    validation_data = CatheterDataset(['validation'], model.type)
    test_data = CatheterDataset(['test'], model.type)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size)
    validation_loader = DataLoader(dataset=validation_data, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size)

    if args.device_ids and use_cuda and len(args.device_ids) > 1:
        model = nn.DataParallel(model, device_ids=[i for i in range(len(args.device_ids))])
    model = model.to(device).apply(weight_init)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=100)
    last_epoch = load_checkpoint(args, model, optimizer, scheduler) if not args.reset else 0

    trainer = Trainer(model, optimizer=optimizer, device=device)

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
        print(f'Train Loss: {train_loss} / Validation Loss: {validation_loss}\n')
    print(f'\nTest Loss: {trainer.test(test_loader)}')

    plt.plot(range(last_epoch + 1, args.epochs + 1), train_losses, label='train loss')
    plt.plot(range(last_epoch + 1, args.epochs + 1), validation_losses, label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=2)

    if args.save_model:
        save_checkpoint('final', args, args.epochs, model, optimizer, scheduler)
        plt.savefig(f'{args.checkpoint_dir}/learning-curve.png', dpi=300)

    if args.visualize:
        plt.show()


main()
