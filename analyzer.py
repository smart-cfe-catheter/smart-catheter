import matplotlib.pyplot as plt
import pandas as pd

record_count = 32
data = pd.DataFrame()

for i in range(record_count):
    record = pd.read_csv(f'data/preprocess/{i + 1}.csv', index_col=False,
                         usecols=['sensor1', 'sensor2', 'sensor3', 'weight'])
    record = record.dropna()
    data = data.append(record, ignore_index=True)

data.hist()
plt.show()
