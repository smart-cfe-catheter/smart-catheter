import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures

from dataset import load_dataset

train, validation, test = load_dataset()
train_x, train_y = train[:][0], train[:][1]
validation_x, validation_y = validation[:][0], validation[:][1]
test_x, test_y = test[:][0], test[:][1]

name = ['Linear Regression', 'Lasso', 'Ridge']
train_scores = [[], [], []]
validation_scores = [[], [], []]
test_scores = [[], [], []]
x_range = range(1, 15)

plt.figure(1)

for degree in x_range:
    transformer = PolynomialFeatures(degree)
    poly_x_train = transformer.fit_transform(train_x)
    poly_x_validation = transformer.fit_transform(validation_x)
    poly_x_test = transformer.fit_transform(test_x)
    print(f"training polynomial regression with degree {degree}...")

    models = [LinearRegression(), Lasso(), Ridge()]
    for idx, model in enumerate(models):
        result = model.fit(poly_x_train, train_y)
        train_scores[idx].append(max(0, result.score(poly_x_train, train_y)))
        validation_scores[idx].append(max(0, result.score(poly_x_validation, validation_y)))
        test_scores[idx].append(max(0, result.score(poly_x_test, test_y)))

        if degree == 3 and idx == 0:
            plt.plot(result.predict(poly_x_test), test_y, ',')
            plt.axis('equal')

for idx in range(len(name)):
    plt.figure(idx + 2)
    plt.title(name[idx])
    plt.xlabel('degree')
    plt.ylabel('R^2 score')

    plt.plot(x_range, train_scores[idx], 'bo', label='train score')
    plt.plot(x_range, validation_scores[idx], 'go', label='validation score')
    plt.plot(x_range, test_scores[idx], 'ro', label='test score')
    plt.legend()

plt.show()
