import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from configs.params import config


class DataLoader:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.scaler = MinMaxScaler()

    def _preprocess_features(self):
        """特征预处理"""
        # 空间特征标准化
        self.df[config.SPATIAL_FEATURES] = self.scaler.fit_transform(
            self.df[config.SPATIAL_FEATURES]
        )

        # 生成时间戳
        self.df['timestamp'] = (self.df['flowStart'] + self.df['flowEnd']) // 2
        self.df.sort_values('timestamp', inplace=True)

    def _create_windows(self):
        """构建时间窗口，确保维度正确"""
        windows, labels = [], []
        n_samples = len(self.df)

        for i in range(0, n_samples - config.WINDOW_SIZE + 1, config.WINDOW_SIZE):
            window = self.df.iloc[i:i + config.WINDOW_SIZE]
            if len(window) < config.WINDOW_SIZE:
                continue

            # 空间特征 (WINDOW_SIZE, len(SPATIAL_FEATURES))
            spatial = window[config.SPATIAL_FEATURES].values

            # 每个窗口的统计特征
            temporal = []
            for feature in config.SPATIAL_FEATURES:
                feature_data = window[feature]
                stats = [
                    feature_data.mean(),
                    feature_data.std(),
                    feature_data.max(),
                    feature_data.min()
                ]
                temporal.extend(stats)

            # 特征处理
            window_features = np.concatenate([
                spatial.reshape(-1),  # 展平空间特征
                np.array(temporal)    # 添加统计特征
            ])

            # 确保特征维度正确
            if len(window_features) != config.FEATURE_DIM:
                # 调整特征维度
                if len(window_features) > config.FEATURE_DIM:
                    window_features = window_features[:config.FEATURE_DIM]
                else:
                    window_features = np.pad(
                        window_features,
                        (0, config.FEATURE_DIM - len(window_features)),
                        'constant'
                    )

            windows.append(window_features)
            labels.append(1 if window['label'].any() else 0)

        return np.array(windows), np.array(labels)

    def load_data(self, test_size=0.3):
        """加载处理后的数据"""
        self._preprocess_features()
        X, y = self._create_windows()

        # 确保X的形状符合要求
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], config.FEATURE_DIM, 1)

        # 时序数据划分（禁止shuffle）
        split_idx = int(len(X) * (1 - test_size))
        return (
            X[:split_idx],
            X[split_idx:],
            y[:split_idx],
            y[split_idx:]
        )