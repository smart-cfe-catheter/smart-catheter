import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PolynomialFeatures

from dataset import CatheterDataset


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

dataset = CatheterDataset()
transformer = PolynomialFeatures(3)

for i in range(3):
    print(f'## Sensor {i}')
    print(f'### Mean: {np.mean(dataset[:][0][:, i])}')
    print(f'### Std: {np.std(dataset[:][0][:, i])}\n')

plt.figure(1)
for i in range(3):
    histogram_plot(dataset, i + 1)

plt.figure(2)
for i in range(3):
    box_plot(dataset, i + 1)

plt.figure(3)
x, y = dataset[:][0], dataset[:][1].reshape(-1)
x = transformer.fit_transform(x)
data = np.hstack((x, np.atleast_2d(y).T))
df = pd.DataFrame(data)
corr = df.corr()
plt.matshow(df.corr())
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()

plt.show()
