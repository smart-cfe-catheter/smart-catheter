import pandas as pd

record_count = 32

for i in range(record_count):
    x_data = pd.read_csv(f'data/input/{i + 1}.csv', index_col=0, header=None,
                         names=['timestamp', 'sensor1', 'sensor2', 'sensor3'], parse_dates=['timestamp'])
    y_data = pd.read_csv(f'data/output/{i + 1}.csv', index_col=0, header=None, names=['timestamp', 'weight'],
                         parse_dates=['timestamp'])

    x_data.index = x_data.index.astype(int)
    y_data.index = y_data.index.astype(int)

    start = 0
    for end in x_data.index:
        start_date = x_data.at[start, 'timestamp']
        end_date = x_data.at[end, 'timestamp']

        if start_date != end_date or end is x_data.index[-1]:
            count = (end - start + 1) if end is x_data.index[-1] else (end - start)
            plus = (end_date - start_date) / count
            for idx in range(count):
                x_data.at[idx + start, 'timestamp'] += plus * idx
            start = end

    data = pd.merge_asof(y_data, x_data, on='timestamp')
    data.to_csv(f'data/preprocess/{i + 1}.csv', index=False, header=True)
    print(f'{i + 1} record preprocess finished')
