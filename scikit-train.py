import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from dataset import load_dataset

parser = argparse.ArgumentParser(description='Smart Catheter Scikit-Trainer')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--degree', type=int, default=15, metavar='S',
                    help='degree of polynomial regression (default: 15)')
parser.add_argument('--save-result', action='store_true', default=False,
                    help='Save results(graphs)')
parser.add_argument('--visualize', action='store_true', default=False,
                    help='Visualize results(graphs)')
args = parser.parse_args()

torch.manual_seed(args.seed)

train, test = load_dataset(no_validation=True)
train_x, train_y = train[:][0], train[:][1].reshape(-1)
test_x, test_y = test[:][0], test[:][1].reshape(-1)

train_losses = []
test_losses = []
x_range = range(1, args.degree)

for degree in x_range:
    transformer = PolynomialFeatures(degree)
    poly_x_train = transformer.fit_transform(train_x)
    poly_x_test = transformer.fit_transform(test_x)
    print(f"training polynomial regression with degree {degree}...")

    model = RidgeCV(alphas=np.arange(0.0, 1.0, 0.1), cv=5)
    result = model.fit(poly_x_train, train_y)
    train_losses.append(mean_squared_error(result.predict(poly_x_train), train_y))
    test_losses.append(mean_squared_error(result.predict(poly_x_test), test_y))

plt.figure(1)
plt.title('Ridge Regression')
plt.xlabel('degree')
plt.ylabel('MSE Loss')

plt.plot(x_range, train_losses, 'bo', label='train score')
plt.plot(x_range, test_losses, 'ro', label='test score')
plt.legend()

print(f'- train: {train_losses}\n- test: {test_losses}')

if args.save_result:
    plt.savefig(f'figures/scikit-learn-ridgecv.png', dpi=300)

if args.visualize:
    plt.show()
