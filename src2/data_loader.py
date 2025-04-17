import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def process_cicids2017(self, data_path):
        """处理CICIDS2017数据集"""
        # 加载数据
        df = pd.read_csv(data_path)

        # 提取特征和标签
        X = df.drop(['Label'], axis=1)  # 假设标签列名为'Label'
        y = df['Label']

        # 处理标签 - 转换为数值编码
        y_encoded = self.label_encoder.fit_transform(y)

        # 获取类别数量
        num_classes = len(self.label_encoder.classes_)
        print(f"类别数量: {num_classes}")
        print("类别映射:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"{label} -> {i}")

        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)

        # 将标签转换为one-hot编码
        y_onehot = to_categorical(y_encoded, num_classes=num_classes)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_onehot,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded  # 保持类别比例
        )

        return X_train, X_test, y_train, y_test, num_classes

    def prepare_data(self, X_train, y_train, X_test, y_test):
        """准备数据为模型所需格式"""
        # 重塑数据为3D格式 (samples, timesteps, features)
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        return X_train_reshaped, y_train, X_test_reshaped, y_test