import matplotlib.pyplot as plt

from dataset import CatheterDataset


def histogram_plot(data, idx):
    plt.subplot(3, 1, idx)
    plt.hist(data.x[:, idx - 1], bins=1000)
    plt.title(f'Sensor{idx}')
    plt.xlabel('\u0394\u03BB')
    plt.ylabel('frequency')


def box_plot(data, idx):
    plt.subplot(1, 3, idx)
    plt.boxplot(data.x[:, idx - 1])
    plt.title(f'Sensor{i + 1}')


train = CatheterDataset()
test = CatheterDataset(train=False)

plt.figure(1)
for i in range(3):
    histogram_plot(train, i + 1)

plt.figure(2)
for i in range(3):
    histogram_plot(test, i + 1)

plt.figure(3)
for i in range(3):
    box_plot(train, i + 1)

plt.figure(4)
for i in range(3):
    box_plot(train, i + 1)

plt.show()
