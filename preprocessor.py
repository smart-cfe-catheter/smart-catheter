import pandas as pd


def get_reference_value(table):
    return table.at[0, 'sensor1'], table.at[0, 'sensor2'], table.at[0, 'sensor3']


record_count = 32
min_val = 1500

for i in range(record_count):
    x_data = pd.read_csv(f'data/input/{i + 1}.csv', index_col=0, header=None,
                         names=['timestamp', 'sensor1', 'sensor2', 'sensor3'], parse_dates=['timestamp'])
    y_data = pd.read_csv(f'data/output/{i + 1}.csv', index_col=0, header=None, names=['timestamp', 'weight'],
                         parse_dates=['timestamp'], dtype={'weight': float})

    x_data, y_data = x_data.dropna().reset_index(drop=True), y_data.dropna().reset_index(drop=True)

    x_data.index = x_data.index.astype(int)
    y_data.index = y_data.index.astype(int)

    start = 0
    for end in x_data.index:
        start_date = x_data.at[start, 'timestamp']
        end_date = x_data.at[end, 'timestamp']

        if x_data.at[end, 'sensor1'] < min_val:
            x_data.at[end, 'sensor1'] *= 1e9
        if x_data.at[end, 'sensor2'] < min_val:
            x_data.at[end, 'sensor2'] *= 1e9
        if x_data.at[end, 'sensor3'] < min_val:
            x_data.at[end, 'sensor3'] *= 1e9

        if start_date != end_date or end is x_data.index[-1]:
            count = (end - start + 1) if end is x_data.index[-1] else (end - start)
            plus = (end_date - start_date) / count
            for idx in range(count):
                x_data.at[idx + start, 'timestamp'] += plus * idx
            start = end

    ref_val1, ref_val2, ref_val3 = get_reference_value(x_data)
    x_data['sensor1'] -= ref_val1
    x_data['sensor2'] -= ref_val2
    x_data['sensor3'] -= ref_val3

    data = pd.merge_asof(y_data, x_data, on='timestamp')
    data['timedelta'] = pd.Timedelta('00:00:00')
    for idx in data.index[-1:0:-1]:
        here = data.at[idx, 'timestamp']
        prev = data.at[idx - 1, 'timestamp']

        data.at[idx, 'timedelta'] = here - prev

    data.to_csv(f'data/preprocess/{i + 1}.csv', index=False, header=True)
    print(f'{i + 1} record preprocess finished')


