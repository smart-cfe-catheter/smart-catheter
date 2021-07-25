import json
import math
import torch
import argparse
import importlib
from tqdm import tqdm
from pathlib import Path
from munch import munchify
from torch.utils.data import DataLoader

from arguments import add_test_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_test_args(parser)
    args = parser.parse_args()

    print('Loading configurations...')
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_file = ckpt_dir / 'best_model.pth'
    ckpt = torch.load(ckpt_file)

    config_file = ckpt_dir / 'args.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = munchify(config)

    model_name = config.model
    model_module = importlib.import_module(f'models.{model_name}')
    model_dataset = getattr(model_module, 'Dataset')

    test_dataset = model_dataset(config, args.test_data)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    model = getattr(model_module, 'Model')(config)
    model.load_state_dict(ckpt['model'])
    model = model.to(args.device)

    with torch.no_grad():
        model.eval()
        total_loss = torch.empty(0).to(args.device)
        for x, y in tqdm(test_loader):
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x)
            pred, y = pred.view(-1), y.view(-1)

            loss = torch.abs(pred - y)
            total_loss = torch.cat((total_loss, loss))

    loss_mean = torch.mean(total_loss).item()
    std_mean = torch.std(total_loss).item()

    print(f'Total Accuracy: {loss_mean}g, 95% Reliability on {1.96 * std_mean / math.sqrt(len(test_dataset))}')
