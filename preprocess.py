import pandas as pd
import numpy as np

total_records = 32

for i in range(total_records):
    x_data = pd.read_csv(f'data/input/{i + 1}.csv', index_col=0, header=None,
                         names=['timestamp', 'sensor1', 'sensor2', 'sensor3'], parse_dates=['timestamp'])
    y_data = pd.read_csv(f'data/output/{i + 1}.csv', index_col=0, header=None, names=['timestamp', 'weight'],
                         parse_dates=['timestamp'])

    ''' 
    interpolation of x_data needed
    '''

    data = pd.merge_asof(y_data, x_data, on='timestamp')
    data.to_csv(f'data/preprocess/{i + 1}.csv', index=False, header=True)
    print(f'{i + 1} record preprocess finished')
