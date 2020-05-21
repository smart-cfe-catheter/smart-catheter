import argparse

import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from dataset import load_dataset

parser = argparse.ArgumentParser(description='Smart Catheter Scikit-Trainer')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--alpha', type=int, default=1, metavar='S',
                    help='alpha value for ridge regression (default: 1)')
parser.add_argument('--degree', type=int, default=15, metavar='S',
                    help='degree of polynomial regression (default: 15)')
parser.add_argument('--save-result', action='store_true', default=False,
                    help='Save results(graphs)')
parser.add_argument('--visualize', action='store_true', default=False,
                    help='Visualize results(graphs)')
args = parser.parse_args()

torch.manual_seed(args.seed)

train, validation, test = load_dataset()
train_x, train_y = train[:][0], train[:][1].reshape(-1)
validation_x, validation_y = validation[:][0], validation[:][1].reshape(-1)
test_x, test_y = test[:][0], test[:][1].reshape(-1)

name = ['Linear Regression', 'Lasso', 'Ridge']
train_losses = [[], [], []]
validation_losses = [[], [], []]
test_losses = [[], [], []]
x_range = range(1, args.degree)

for degree in x_range:
    transformer = PolynomialFeatures(degree)
    poly_x_train = transformer.fit_transform(train_x)
    poly_x_validation = transformer.fit_transform(validation_x)
    poly_x_test = transformer.fit_transform(test_x)
    print(f"training polynomial regression with degree {degree}...")

    models = [LinearRegression(), Lasso(), Ridge(alpha=args.alpha)]
    for idx, model in enumerate(models):
        result = model.fit(poly_x_train, train_y)
        train_losses[idx].append(mean_squared_error(result.predict(poly_x_train), train_y))
        validation_losses[idx].append(mean_squared_error(result.predict(poly_x_validation), validation_y))
        test_losses[idx].append(mean_squared_error(result.predict(poly_x_test), test_y))

for idx in range(len(name)):
    plt.figure(idx + 1)
    plt.title(name[idx])
    plt.xlabel('degree')
    plt.ylabel('MSE Loss')

    plt.plot(x_range, train_losses[idx], 'bo', label='train score')
    plt.plot(x_range, validation_losses[idx], 'go', label='validation score')
    plt.plot(x_range, test_losses[idx], 'ro', label='test score')
    plt.legend()

    print(f'\n<{name[idx]}>\n'
          f'- train: {train_losses[idx]}\n'
          f'- validation: {validation_losses[idx]}\n'
          f'- test: {test_losses[idx]}')

    if args.save_result:
        plt.savefig(f'figures/scikit-learn-{name[idx]}.png', dpi=300)

if args.visualize:
    plt.show()
