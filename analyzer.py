import argparse

import matplotlib.pyplot as plt
import torch

from dataset import load_dataset


def histogram_plot(data, idx):
    plt.subplot(3, 1, idx)
    plt.hist(data[:][0][:, idx - 1], bins=1000)
    plt.title(f'Sensor{idx}')
    plt.xlabel('\u0394\u03BB')
    plt.ylabel('frequency')


def box_plot(data, idx):
    plt.subplot(1, 3, idx)
    plt.boxplot(data[:][0][:, idx - 1])
    plt.title(f'Sensor{i + 1}')


parser = argparse.ArgumentParser(description='Smart Catheter Trainer')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

torch.manual_seed(args.seed)

datasets = load_dataset()
titles = ['Train', 'Validation', 'Test']

for idx, dataset in enumerate(datasets):
    plt.figure(2 * idx + 1)
    for i in range(3):
        histogram_plot(dataset, i + 1)

    plt.figure(2 * idx + 2)
    for i in range(3):
        box_plot(dataset, i + 1)

plt.show()
