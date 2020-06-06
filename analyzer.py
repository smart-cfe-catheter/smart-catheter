import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import CatheterDataset


def histogram_plot(data, title, label):
    plt.title(title)
    plt.hist(data, bins=1000)
    plt.xlabel(label)


def box_plot(data, title):
    plt.boxplot(data, vert=False)
    plt.title(title)


def main():
    torch.manual_seed(1)
    dataset = CatheterDataset(folders=['train', 'validation', 'test'])
    x_data, y_data = dataset[:][0].numpy(), dataset[:][1].numpy()

    for i in range(3):
        x = x_data[:, i]
        print(f'<Sensor {i + 1}>')
        print(f'- Mean: {np.mean(x)}')
        print(f'- Std: {np.std(x)}')
        print(f'- Median: {np.median(x)}')

    plt.figure(1)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        histogram_plot(x_data[:, i], f'Sensor{i + 1}', '\u0394\u03BB')

    plt.figure(2)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        box_plot(x_data[:, i], f'Sensor{i + 1}')

    plt.figure(3)
    plt.subplot(2, 1, 1)
    histogram_plot(y_data, 'Weight', 'g')
    plt.subplot(2, 1, 2)
    box_plot(y_data, 'Weight')

    plt.show()


main()
