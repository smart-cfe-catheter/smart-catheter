from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

from formatter import formatter

data = formatter(32, time_series=False, ignore_nan=True)

x_data = data.loc[:, ['sensor1', 'sensor2', 'sensor3']].copy()
y_data = data.loc[:, ['weight']].copy()

x_train, x_test, y_train, y_test = train_test_split(x_data.values, y_data.values, random_state=77)

name = ['Linear Regression', 'Lasso', 'Ridge']
train_scores = [[], [], []]
test_scores = [[], [], []]
x_range = range(1, 15)

plt.figure(1)

for degree in x_range:
    transformer = PolynomialFeatures(degree)
    poly_x_train = transformer.fit_transform(x_train)
    poly_x_test = transformer.fit_transform(x_test)
    print(f"training polynomial regression with degree {degree}...")

    models = [LinearRegression(), Lasso(), Ridge()]
    for idx, model in enumerate(models):
        result = model.fit(poly_x_train, y_train)
        train_scores[idx].append(max(0, result.score(poly_x_train, y_train)))
        test_scores[idx].append(max(0, result.score(poly_x_test, y_test)))

        if degree == 3 and idx == 0:
            plt.plot(result.predict(poly_x_test), y_test, ',')
            plt.axis('equal')

for idx in range(len(name)):
    plt.figure(idx + 2)
    plt.title(name[idx])
    plt.xlabel('degree')
    plt.ylabel('R^2 score')

    plt.plot(x_range, train_scores[idx], 'bo', label='train score')
    plt.plot(x_range, test_scores[idx], 'ro', label='test score')
    plt.legend()

plt.show()
