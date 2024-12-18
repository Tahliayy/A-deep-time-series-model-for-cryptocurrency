class ModelParams:
    hidden_layer_size = 12                              # originally, choose from [5, 10, 20, 40, 80, 160]
    dropout_rate = 0.1                                  # originally, choose from [0.1, 0.2, 0.3, 0.4, 0.5]
    max_gradient_norm = 1.0                             # originally, choose from [0.01, 1.0, 100.0]
    lr = 0.001                                          # originally, choose from [1e-4, 1e-3, 1e-2, 1e-1]
    min_lr = 1e-5

    # feature dimensions
    num_features = 42
    category_dim = 1
    category_cnt = 19

    # tft embedding
    embedding_hidden_layer_size = hidden_layer_size

    # multi head attention
    num_heads = 3


class params:
    """
    We use this class to set parameters using in all areas
    """
    # device choice
    use_cuda = True                                    # True: by default,     False: usable. for debugging as well

    # data files
    data_root = 'data/12h'

    # training params
    batch_size = 19
    seq_len_per_batch_train = 60
    seq_len_per_batch_predict = 60
    train_window = seq_len_per_batch_train * 120        # we use 120 windows to train where each window's seq_len = 60
    predict_window = seq_len_per_batch_predict * 12     # then we test 12 windows where each window's seq_len = 60

    epoch_per_window = 35                               # training epoch times per window

    # model
    model_params = ModelParams()

    # saving folder
    output_dir = 'tmt_results'





