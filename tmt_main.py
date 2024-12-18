import os
import time
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from model import tmt_model
from tmt_data import get_data_list, get_windows
from tmt_config import params
from tmt_util import id_ticker_dict


def get_batch(src_data, start_pos, seq_len):
    in_data = src_data[:,                                                       # batch
                       start_pos:start_pos + seq_len,         # seq_len
                       :params.model_params.num_features]                       # feat_dim = 0~41
    in_ids = src_data[:,
                      start_pos:start_pos + seq_len,
                      -2]                                                       # id = 42
    ground_truth = src_data[:,
                            start_pos:start_pos + seq_len,
                            -1]                                                 # ground_truth = 43
    return in_data, in_ids, ground_truth


def train_loop(dev, training_windows):
    print('====> Training Loop')
    pred_dict, act_dict = defaultdict(list), defaultdict(list)
    cal_pred_dict, cal_act_dict = defaultdict(list), defaultdict(dict)

    # settings: model
    model = tmt_model.TMT(device=dev).to(dev)

    # settings: loss
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=params.model_params.lr)   # 0.001
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # training loop, training_windows: 251 x {[19, 7200, 44], [19, 720, 44]}
    for window_id, window in enumerate(training_windows):
        # training_data: np.array(19, 7200, 44), features: 0-41, id: 42, target: 43
        # testing_data: np.array(19, 720, 44)
        training_data, testing_data = window

        print('window_id: ', window_id + 1)
        start_time = time.time()

        # train
        model.train()
        # epoch_loss = 0
        for epoch_id in range(params.epoch_per_window):
            for batch_start in range(0, training_data.shape[1] - 1, params.seq_len_per_batch_train):
                # features: (19, 60, 42)
                # ids: (19, 60) -> [19, 60, 1]
                # targets: [19, 60]
                features, ids, targets = \
                    get_batch(training_data, batch_start, params.seq_len_per_batch_train)
                train_features = torch.tensor(features).float().to(dev)
                train_ids = torch.tensor(ids).int().unsqueeze(-1).to(dev)
                train_targets = torch.tensor(targets).float().to(dev)

                optimizer.zero_grad()
                train_outputs = model(train_features, train_ids, 'train')         # [19, 60]
                loss = 5 * l1_loss(train_outputs, train_targets)
                # epoch_loss += loss.item()
                if dev.type == 'cpu':
                    # the computing would be super slow, so show this to prove it's running.
                    # but 'cpu' mode is definitely not recommended
                    print(f'     window_id {window_id}, train loss {epoch_id}: {loss.item()}')
                loss.backward()
                optimizer.step()
            # print('window')

        # update learning rate, in case it's too large for larger window_id
        scheduler.step()
        for param_group in optimizer.param_groups:
            if param_group['lr'] < params.model_params.model.min_lr:
                param_group['lr'] = params.model_params.model.min_lr

        # test / predict
        model.eval()
        with torch.no_grad():
            for epoch_id in range(params.epoch_per_window):
                for pred_batch_start in range(0, testing_data.shape[1] - 1, params.seq_len_per_batch_predict):
                    features, ids, targets = \
                        get_batch(testing_data, pred_batch_start, params.seq_len_per_batch_predict)
                    pred_features = torch.tensor(features).float().to(dev)
                    pred_ids = torch.tensor(ids).int().unsqueeze(-1).to(dev)
                    pred_targets = torch.tensor(targets).float().to(dev)            # [19, 60]

                    pred_outputs = model(pred_features, pred_ids, 'pred')           # [19, 60]

                    # 1. calculate the average loss over the sequence for each ticker and obtain the loss_list
                    #    loss for each ticker's seq
                    batch_loss = torch.mean((pred_outputs - pred_targets) ** 2, dim=1)  # [19]
                    # loss_list = batch_loss.tolist()  # convert to list, used for presenting

                    # 2. calcualte the total loss (mean loss over all tickers)
                    batch_loss = batch_loss.mean().item()
                    print(f'     window_id: {window_id}, eval loss:  {batch_loss}')

                    # 3. convert pred_outputs and pred_targets to dictionary
                    for i in range(pred_outputs.shape[0]):  # traverse each ticker within the batch
                        ticker_name = id_ticker_dict[i]
                        # unroll the predicted and target values for each ticker and convert them into lists
                        pred_sequence = pred_outputs[i].view(-1).tolist()  # [60] 展开后转换为列表
                        target_sequence = pred_targets[i].view(-1).tolist()  # [60]

                        # 4. 扩展每个 ticker 的预测和真实值（多个批次）
                        pred_dict[ticker_name].extend(pred_sequence)  # 将当前 batch 的预测值追加到对应 ticker 的列表中
                        act_dict[ticker_name].extend(target_sequence)  # 将当前 batch 的目标值追加到对应 ticker 的列表中

                        # this is for calculation
                        if window_id >= 201:
                            cal_pred_dict[ticker_name].extend(pred_sequence)
                            cal_act_dict[ticker_name].extend(target_sequence)

        end_time = time.time()

        print('     iteration time in seconds: ', end_time - start_time)

    return pred_dict, act_dict, cal_pred_dict, cal_act_dict


def post_process(pred_output_dict, act_output_dict, cal_pred_values_dict, cal_act_values_dict):
    """
    We use this function to show the predicting results
    :param pred_output_dict: dict: key-ticker, values: predicted results
    :param act_output_dict: dict: key-ticker, values: real outputs aligning with predicted results
    :param cal_pred_values_dict: dict, the latter part of pred_output, used for computing the loss
    :param cal_act_values_dict: dict, the latter par of act_output, used for computing the loss
    :return: None
    """
    # create saving folder
    os.makedirs(params.output_dir, exist_ok=True)

    # iterate tickers
    cnt, total_mse = 0, 0
    for ticker_name in pred_output_dict:
        predicted_values = pred_output_dict[ticker_name]
        actual_values = act_output_dict[ticker_name]

        # this is for plotting
        results_df = pd.DataFrame({
            'Actual': actual_values,
            'Predicted': predicted_values
        })
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['Actual'], label='Actual', color='blue')
        plt.plot(results_df['Predicted'], label='Predicted', color='red', linestyle='dashed')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title(f'Actual vs Predicted Values for {ticker_name}')
        plt.legend()

        # this is for saving
        plt.savefig(os.path.join(params.output_dir, f'{ticker_name}_plot.png'))
        plt.close()

        # this is for calculating
        cal_predicted_values = cal_pred_values_dict[ticker_name]
        cal_actual_values = cal_act_values_dict[ticker_name]
        cal_results_df = pd.DataFrame({
            'Actual': cal_actual_values,
            'Predicted': cal_predicted_values
        })
        mse = np.mean((cal_results_df['Actual'] - cal_results_df['Predicted']) ** 2)
        print('ticker_name: ', ticker_name, '  mse: ', mse)

        total_mse += mse
        cnt += 1

    print('avg mse: ', total_mse / cnt)


def main():
    # get running device, GPU by default
    device = torch.device('cuda' if torch.cuda.is_available() and params.use_cuda else 'cpu')

    # extract data info,              data_root = 'data/12h'
    data_array = get_data_list(params.data_root)     # np.array(19, 187921, 44), features: 0-41, id: 42, target: 43
    data_windows = get_windows(data_array)           # 25 x {[19, 7200, 44], [19, 720, 44]}

    # main training loop where we train the model, test the model
    predicted_outputs, actual_outputs, cal_predicted_values, cal_actual_values = train_loop(device, data_windows)

    # evaluation, illustration, etc
    post_process(predicted_outputs, actual_outputs, cal_predicted_values, cal_actual_values)

    a = 1


if __name__ == '__main__':
    main()


# # Save the predictions to a DataFrame
# results_df = pd.DataFrame({
#     'Actual': actual_values,
#     'Predicted': predicted_values
# })
#
# cal_results_df = pd.DataFrame({
#     'Actual': cal_actual_values,
#     'Predicted': cal_predicted_values
# })
#
# mse = np.mean((cal_results_df['Actual'] - cal_results_df['Predicted'])**2)
# print('mse: ', mse)
#
# # 假设 results_df 是你的 DataFrame
# with open('new_main.pickle', 'wb') as f:
#     pickle.dump(results_df, f)
#
# # Plot the actual vs predicted values
# plt.figure(figsize=(12, 6))
# #plt.plot(x, label='Actual', color='red')
# plt.plot(results_df['Actual'], label='Actual', color='blue')
# plt.plot(results_df['Predicted'], label='Predicted', color='red', linestyle='dashed')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.title('Actual vs Predicted Values')
# plt.legend()
# plt.show()










