import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

record_count = 32
data = pd.DataFrame()

for i in range(record_count):
    record = pd.read_csv(f'data/preprocess/{i + 1}.csv', index_col=False,
                         usecols=['sensor1', 'sensor2', 'sensor3', 'weight'])
    record = record.dropna()
    data = data.append(record, ignore_index=True)

x_data = data.loc[:, ['sensor1', 'sensor2', 'sensor3']].copy()
y_data = data.loc[:, ['weight']].copy()

x_train, x_test, y_train, y_test = train_test_split(x_data.values, y_data.values, random_state=77)

model = LinearRegression().fit(x_train, y_train)

print(f"train score : {model.score(x_train, y_train)}")
print(f"test score: {model.score(x_test, y_test)}")
