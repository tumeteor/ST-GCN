class TGCN:
    max_nb_epochs = 45
    input_dim = 29
    hidden_dim = 64
    layer_dim = 2
    output_dim = 1
    train_percent_check = 1
    lr = 0.01
    weight_decay = 0.015
    lr_scheduler_patience = 3


class Data:
    train_num_steps = 251
    valid_num_steps = 51
    test_num_steps = 11
