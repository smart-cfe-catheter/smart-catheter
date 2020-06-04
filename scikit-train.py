import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from dataset import load_dataset

parser = argparse.ArgumentParser(description='Smart Catheter Scikit-Trainer')
parser.add_argument('--degree', type=int, default=5)
parser.add_argument('--save-result', action='store_true', default=False)
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--model', type=str, default='Ridge', choices=['Lasso', 'Ridge'])
args = parser.parse_args()

torch.manual_seed(1)

train, test = load_dataset(no_validation=True)
train_x, train_y = train[:][0], train[:][1].reshape(-1)
test_x, test_y = test[:][0], test[:][1].reshape(-1)

train_losses = []
test_losses = []
xrange = range(1, args.degree + 1)

for degree in xrange:
    transformer = PolynomialFeatures(degree)
    poly_x_train = transformer.fit_transform(train_x)
    poly_x_test = transformer.fit_transform(test_x)

    alphas = np.arange(0.0, 1.0, 0.1) if args.model == 'Ridge' \
        else np.arange(0.1, 1.0, 0.1)
    model = RidgeCV(alphas=alphas, cv=5) if args.model == 'Ridge' \
        else LassoCV(n_alphas=len(alphas), alphas=alphas, cv=5)
    result = model.fit(poly_x_train, train_y)
    train_loss = mean_squared_error(result.predict(poly_x_train), train_y)
    test_loss = mean_squared_error(result.predict(poly_x_test), test_y)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Degree #{degree} Train loss: {train_loss} / Test Loss: {test_loss}")

plt.figure(1)
plt.title(f'{args.model} Regression')
plt.xlabel('degree')
plt.ylabel('MSE Loss [N]')

plt.plot(xrange, train_losses, 'bo', label='train score')
plt.plot(xrange, test_losses, 'ro', label='test score')
plt.legend()

print(f'- Train Loss: {train_losses}\n- Test Loss: {test_losses}')

if args.save_result:
    plt.savefig(f'results/{args.model}.png', dpi=300)

if args.visualize:
    plt.show()
