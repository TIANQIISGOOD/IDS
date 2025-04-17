import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from params import config

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()

    def load_data(self):
        # 读取CSV文件
        df = pd.read_csv(self.file_path)

        # 分离特征和标签
        X = df[config.FEATURE_COLUMNS].values
        y = df[config.LABEL_COLUMN].values

        # 标准化特征
        X = self.scaler.fit_transform(X)

        # 将标签转换为one-hot编码
        y = to_categorical(y, num_classes=config.NUM_CLASSES)

        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test