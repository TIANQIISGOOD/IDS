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

    # # 模型参数
    # CNN_FILTERS = 32  # 卷积核数量
    # LSTM_UNITS = 64  # LSTM单元数
    # DROPOUT_RATE = 0.3  # Dropout比率
    # REGULARIZATION = 0.01  # L2正则化系数
    #
    # # 训练参数
    # BATCH_SIZE = 32
    # INIT_LR = 0.001
    # EPOCHS = 50
    # CLASS_WEIGHTS = {0: 1, 1: 5}  # 类别权重

    # 改进的模型参数
    CNN_FILTERS = 64  # 增加卷积核数量
    BiLSTM_UNITS = 128  # 增加LSTM单元数
    DROPOUT_RATE = 0.25  # 调整Dropout率
    REGULARIZATION = 0.001  # 降低正则化系数

    # 改进的训练参数
    BATCH_SIZE = 32  # 增加批次大小
    INIT_LR = 0.001  # 降低初始学习率
    EPOCHS = 100  # 增加训练轮数
    CLASS_WEIGHTS = {0: 1, 1: 5}  # 增加正样本权重
    # 特征维度配置
    FEATURE_DIM = 102  # 根据实际计算值调整

    # 模型对比配置
    COMPARE_MODELS = ['BiLSTM-CNN', 'CNN', 'BiLSTM', 'RandomForest']
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'class_weight': 'balanced'
    }


config = Config()