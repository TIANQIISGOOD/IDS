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
        """构建时间窗口（修复维度问题）"""
        windows, labels = [], []
        n_samples = len(self.df)

        for i in range(0, n_samples, config.WINDOW_SIZE):
            window = self.df.iloc[i:i + config.WINDOW_SIZE]
            if len(window) < config.WINDOW_SIZE:
                continue

            # 空间特征（保持二维结构）
            spatial = window[config.SPATIAL_FEATURES].values  # shape: (30,3)

            # 时序统计量（调整为二维结构）
            temporal = window[config.SPATIAL_FEATURES].agg(
                ['mean', 'var', 'max', 'min']
            ).T.values.reshape(-1, 1)  # shape: (12,1)

            # 特征融合（使用堆叠代替拼接）
            window_features = np.hstack([
                spatial.flatten(),  # 将空间特征展平为一维
                temporal.flatten()  # 将时序统计量展平为一维
            ])

            windows.append(window_features)
            labels.append(1 if window['label'].any() else 0)

        return np.array(windows), np.array(labels)


    def load_data(self, test_size=0.3):
        """加载处理后的数据"""
        self._preprocess_features()
        X, y = self._create_windows()

        # 时序数据划分（禁止shuffle）
        split_idx = int(len(X) * (1 - test_size))
        return (
            X[:split_idx],
            X[split_idx:],
            y[:split_idx],
            y[split_idx:]
        )