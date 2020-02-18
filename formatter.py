import pandas as pd


def formatter(record_count, time_series=False, ignore_nan=False):
    data = pd.DataFrame()

    for i in range(record_count):
        record = pd.read_csv(f'data/preprocess/{i + 1}.csv', index_col=False,
                             usecols=['sensor1', 'sensor2', 'sensor3', 'weight'])
        if ignore_nan:
            record = record.dropna()
        if not time_series:
            data = data.append(record, ignore_index=True)

    return data


