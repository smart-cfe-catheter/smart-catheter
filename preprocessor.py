import random

import numpy as np
import pandas as pd
import pywt
from scipy.interpolate import interp1d

from preprocess import frequency


def save_to_file(x, y, prefix):
    print(f'{prefix} {len(x)}')
    x, y = np.stack(x, axis=0), np.stack(y, axis=0)
    x, y = np.float32(x), np.float32(y)
    
    fx, fy = open(f'{prefix}_signal.in', 'wb'), open(f'{prefix}_scale.in', 'wb')
    x, y = x.tobytes(), y.tobytes()
    fx.write(x), fy.write(y)

    fx.close(), fy.close()


def save_data(x, y, root_dir):
    total_cnt = len(x)

    train_cnt = int(total_cnt * 0.9)
    val_cnt = int(train_cnt * 0.2)

    x_train, y_train = x[0:(train_cnt - val_cnt)], y[0:(train_cnt - val_cnt)]
    x_val, y_val = x[(train_cnt - val_cnt):train_cnt], y[(train_cnt - val_cnt):train_cnt]
    x_test, y_test = x[train_cnt:], y[train_cnt:]

    save_to_file(x_train, y_train, f'{root_dir}/train')
    save_to_file(x_val, y_val, f'{root_dir}/validation')
    save_to_file(x_test, y_test, f'{root_dir}/test')


def cal_time_delta(frame):
    frame['timedelta'] = (frame['timestamp'] - frame.at[0, 'timestamp']).dt.total_seconds()
    frame = frame.drop(['timestamp'], axis=1)
    return frame


def interpolate(data, freq, xrange):
    interpolator = interp1d(freq, data, axis=0)
    return interpolator(xrange)


def create_window_spectrogram(x, y):
    new_x, new_y = np.empty((0, frequency, frequency, 3)), np.empty((0, 1))

    for start in range(1, x.shape[0] - frequency, frequency):
        end = start + frequency
        x_window = x[start:end] - x[end - 1]
        x_spectrogram = np.empty((frequency, frequency, 3))
        for channel in range(3):
            result, _ = pywt.cwt(x_window[:, channel], range(1, frequency + 1), 'morl', 0.001)
            x_spectrogram[:, :, channel] = result

        y_window = y[end - 1] - y[start - 1]
        y_window = y_window.reshape((-1, 1))

        new_x = np.append(new_x, x_spectrogram.reshape((-1, frequency, frequency, 3)), axis=0)
        new_y = np.append(new_y, y_window, axis=0)
    return new_x.reshape((-1, frequency * frequency * 3)), new_y


random.seed(1)
record_count = 130
max_len = [0, 0]
ban_list = [11, 18, 19, 28, 32, 36, 37, 38, 39, 40, 41, 42, 45, 48, 53, 54, 123]
x_data_list = [[], []]
y_data_list = [[], []]

for i in range(1, record_count + 1):
    if i in ban_list:
        continue
    x_data = pd.read_csv(f'data/raw/input/{i}.csv', index_col=0, header=None,
                         names=['timestamp', 'sensor1', 'sensor2', 'sensor3'], parse_dates=['timestamp'])
    y_data = pd.read_csv(f'data/raw/output/{i}.csv', index_col=0, header=None, names=['timestamp', 'weight'],
                         parse_dates=['timestamp'], dtype={'weight': float})

    x_data, y_data = x_data.dropna().reset_index(drop=True), y_data.dropna().reset_index(drop=True)
    if i < 55:
        x_data = x_data[x_data['sensor3'] > 1538]
    x_data, y_data = cal_time_delta(x_data), cal_time_delta(y_data)
    x_data, y_data = x_data.to_numpy(), y_data.to_numpy()
    x_data[:, [0, 1, 2]] -= x_data[0, [0, 1, 2]]

    xrange = np.arange(0, min(x_data[-1, 3], y_data[-1, 1]), 0.001)
    x_data = interpolate(x_data[:, [0, 1, 2]], x_data[:, 3], xrange)
    y_data = interpolate(y_data[:, [0]], y_data[:, 1], xrange)
    max_len[0] = max(max_len[0], x_data.shape[0])
    x_data_list[0].append(x_data)
    y_data_list[0].append(y_data)

    x_data, y_data = create_window_spectrogram(x_data, y_data)
    max_len[1] = max(max_len[1], x_data.shape[0])
    x_data_list[1].append(x_data)
    y_data_list[1].append(y_data)

    print(f'Record #{i} finished preprocess')

for i in range(0, 2):
    for j in range(len(x_data_list[i])):
        length = x_data_list[i][j].shape[0]

        x_data_list[i][j] = np.pad(x_data_list[i][j], ((max_len[i] - length, 0), (0, 0)), mode='constant', constant_values=0.0)
        y_data_list[i][j] = np.pad(y_data_list[i][j], ((max_len[i] - length, 0), (0, 0)), mode='constant', constant_values=0.0)
        
save_data(x_data_list[0], y_data_list[0], './data/preprocess/series')
save_data(x_data_list[1], y_data_list[1], './data/preprocess/spectrogram')
print('finished preprocess')

