import os

import numpy as np
import pandas as pd

from tmt_util import ticker_id_dict
from tmt_config import params


def get_data_list(data_root):
    print('====> Read Data')
    ticker_data_list = list()
    ticker_file_name_list = os.listdir(data_root)
    for ticker_file_name in ticker_file_name_list:
        # deal with ticker name
        ticker_name = ticker_file_name.split('.')[0].split('_')[0]
        if ticker_name == 'AVAX':
            continue

        # read ticker data
        ticker_file = os.path.join(data_root, ticker_file_name)
        ticker_data = pd.read_csv(ticker_file)

        # extract useful info
        target_column_id = 'target_12h_lprice'
        ticker_features = ticker_data.drop(columns=['Unnamed: 0', target_column_id]).values  # [187921, 42]
        ticker_target = ticker_data[target_column_id].values  # [187921,]
        ticker_id = ticker_id_dict[ticker_name]     # {int}

        ticker_data_combined = np.hstack((ticker_features,
                                          np.full((ticker_features.shape[0], 1), ticker_id),
                                          ticker_target.reshape(-1, 1)))

        # merge data together
        ticker_data_list.append(ticker_data_combined)

    ticker_data_array = np.stack(ticker_data_list, axis=0)

    return ticker_data_array


def get_windows(all_ticker_data):
    """
    This function is used for getting training loop windows.
    :param all_ticker_data: [19, 187921, 44]
    :return: training windows: 25 x {[19, 7200, 44], [19, 720, 44]}
    """
    print('====> Get training loop windows')
    windows = list()
    for start in range(0,
                       all_ticker_data.shape[1] - params.train_window - params.predict_window + 1,
                       params.predict_window):
        end = start + params.train_window
        predict_end = end + params.predict_window
        train_data = all_ticker_data[:, start:end, :]
        test_data = all_ticker_data[:, end:predict_end, :]
        windows.append((train_data, test_data))

    return windows
