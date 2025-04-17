class Config:
    # 数据参数
    FEATURE_DIM = 26  # 更新为实际特征数量（不包含标签列）
    NUM_CLASSES = 8  # 更新为8个类别

    # 模型参数
    CNN_FILTERS = 64
    BiLSTM_UNITS = 128
    DROPOUT_RATE = 0.25
    REGULARIZATION = 0.001

    # 训练参数
    BATCH_SIZE = 32
    INIT_LR = 0.001
    EPOCHS = 100

    # 特征列名
    FEATURE_COLUMNS = [
        'command_address', 'response_address', 'command_memory',
        'response_memory', 'command_memory_count', 'response_memory_count',
        'comm_read_function', 'comm_write_fun', 'resp_read_fun',
        'resp_write_fun', 'sub_function', 'command_length',
        'resp_length', 'gain', 'reset', 'deadband',
        'cycletime', 'rate', 'setpoint', 'control_mode',
        'control_scheme', 'pump', 'solenoid', 'crc_rate',
        'measurement', 'time'
    ]

    # 标签列名
    LABEL_COLUMN = 'result'


config = Config()