import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def get_reference_value(table):
    return table.at[0, 'sensor1'], table.at[0, 'sensor2'], table.at[0, 'sensor3']


def mad_outlier_detection(values, mad, median):
    new_values = values.copy()
    new_values[(0.6745 * (values - median)).abs() > 10.0 * mad] = np.nan
    return new_values


record_count = 89
min_val = 1500
ban_list = [32, 33, 57, 64, 86]
global_cnt = 1

for i in range(record_count):
    if i in ban_list:
        continue
    x_data = pd.read_csv(f'data/input/{i + 1}.csv', index_col=0, header=None,
                         names=['timestamp', 'sensor1', 'sensor2', 'sensor3'], parse_dates=['timestamp'])
    y_data = pd.read_csv(f'data/output/{i + 1}.csv', index_col=0, header=None, names=['timestamp', 'weight'],
                         parse_dates=['timestamp'], dtype={'weight': float})

    x_data.index = x_data.index.astype(int)
    y_data.index = y_data.index.astype(int)
    ref_val1, ref_val2, ref_val3 = get_reference_value(x_data)

    x_data['sensor1'] -= ref_val1
    x_data['sensor2'] -= ref_val2
    x_data['sensor3'] -= ref_val3
    x_data['sensor3'] = mad_outlier_detection(x_data['sensor3'], 0.03910100000007333, -0.0049409999999170395)

    x_data, y_data = x_data.dropna().reset_index(drop=True), y_data.dropna().reset_index(drop=True)

    raw_data = pd.merge_asof(y_data, x_data, on='timestamp', direction='nearest', tolerance=pd.Timedelta('5ms'))
    raw_data = raw_data.dropna().reset_index(drop=True)
    raw_data['timedelta'] = 0.0
    for idx in raw_data.index[-1:0:-1]:
        here = raw_data.at[idx, 'timestamp']
        prev = raw_data.at[idx - 1, 'timestamp']

        raw_data.at[idx, 'timedelta'] = (here - prev).total_seconds()

    raw_data = raw_data.drop(['timestamp'], axis=1)
    data = raw_data.to_numpy()
    data = gaussian_filter1d(data, 2, axis=0)

    np.savetxt(f'data/preprocess/{global_cnt}.csv', data, delimiter=',')
    print(f'{global_cnt} record preprocess finished')
    global_cnt += 1
