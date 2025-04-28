class Config:
    # 数据参数
    WINDOW_SIZE = 30
    FEATURE_DIM = 25  # 更新为实际数值特征数量（原32列移除3个IP相关列）
    NUM_CLASSES = 2

    # 特征列名（移除src_ip, dst_ip, proto等字符串列）
    FEATURE_COLUMNS = [
        'src_port', 'dst_port',
        'pktTotalCount', 'octetTotalCount', 'avg_ps',
        'service_errors', 'status_errors', 'msg_size',
        'min_msg_size', 'flowStart', 'flowEnd', 'flowDuration',
        'avg_flowDuration', 'flowInterval', 'count', 'srv_count',
        'same_srv_rate', 'dst_host_same_src_port_rate', 'f_pktTotalCount',
        'f_octetTotalCount', 'f_flowStart', 'f_rate', 'b_pktTotalCount',
        'b_octetTotalCount', 'b_flowStart'
    ]

    # 标签列名
    LABEL_COLUMN = 'label'  # 使用多分类标签列
    LABEL_MAPPING = {
        "Normal": 0,
        "DoS": 1,
        "Impersonation": 2,
        "MITM": 3
    }

    # 保持原模型参数
    CNN_FILTERS = 64
    LSTM_UNITS = 128
    DROPOUT_RATE = 0.25
    REGULARIZATION = 0.001

    # 保持原训练参数
    BATCH_SIZE = 128
    INIT_LR = 0.001
    EPOCHS = 20
    CLASS_WEIGHTS = None

config = Config()