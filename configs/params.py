class Config:
    # 数据参数
    WINDOW_SIZE = 30  # 时间窗口大小（秒）
    SPATIAL_FEATURES = [  # 空间特征选择
        'avg_ps',
        'pktTotalCount',
        'flowInterval'
    ]
    TEMPORAL_FEATURES = [
        'flowStart',
        'flowEnd'
    ]

    # 模型参数
    CNN_FILTERS = 64  # 卷积核数量
    LSTM_UNITS = 128  # LSTM单元数
    DROPOUT_RATE = 0.25  # Dropout率
    REGULARIZATION = 0.001  # 正则化系数

    # 训练参数
    BATCH_SIZE = 32
    INIT_LR = 0.001  # 初始学习率
    EPOCHS = 100
    CLASS_WEIGHTS = {0: 1, 1: 5}  # 类别权重

    # 特征维度配置
    FEATURE_DIM = 102  # 特征维度

    # 模型对比配置
    COMPARE_MODELS = ['BiLSTM-CNN', 'CNN', 'BiLSTM', 'RandomForest']
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'class_weight': 'balanced'
    }


config = Config()