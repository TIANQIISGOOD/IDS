class Config:
    # 数据参数
    WINDOW_SIZE = 30  # 时间窗口大小（秒）
    FEATURE_DIM = 26  # 特征维度
    NUM_CLASSES = 2  # 类别数量

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

    # 模型参数
    CNN_FILTERS = 64
    LSTM_UNITS = 128
    DROPOUT_RATE = 0.25
    REGULARIZATION = 0.001

    # 训练参数
    BATCH_SIZE = 128
    INIT_LR = 0.001
    EPOCHS = 1

    # 类别权重（根据数据分布调整）
    CLASS_WEIGHTS = None  # 训练时会根据数据分布自动计算


config = Config()