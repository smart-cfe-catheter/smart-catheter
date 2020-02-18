from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from formatter import formatter

data = formatter(32, time_series=False, ignore_nan=True)

x_data = data.loc[:, ['sensor1', 'sensor2', 'sensor3']].copy()
y_data = data.loc[:, ['weight']].copy()

x_train, x_test, y_train, y_test = train_test_split(x_data.values, y_data.values, random_state=77)

model = LinearRegression().fit(x_train, y_train)

print(f"train score : {model.score(x_train, y_train)}")
print(f"test score: {model.score(x_test, y_test)}")
