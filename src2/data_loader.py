import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from params import config


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()

    def load_data(self):
        # 读取原始数据
        df = pd.read_csv(self.file_path)

        # ========== 关键修改1：移除非数值列 ==========
        drop_columns = ['src_ip', 'dst_ip', 'proto', 'flags', 'service']
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])

        # ========== 关键修改2：处理时间戳 ==========
        for col in ['flowStart', 'flowEnd', 'f_flowStart', 'b_flowStart']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).astype('int64') // 10 ** 9  # 转换为Unix时间戳

        # ========== 关键修改3：标签映射 ==========
        # # 检查标签列是否存在
        # if config.LABEL_COLUMN not in df.columns:
        #     raise ValueError(f"标签列 '{config.LABEL_COLUMN}' 不存在")
        #
        # # 映射文本标签到数字
        # y = df[config.LABEL_COLUMN].map(config.LABEL_MAPPING)
        #
        # # 检查未定义标签
        # if y.isna().any():
        #     invalid_labels = df.loc[y.isna(), config.LABEL_COLUMN].unique()
        #     raise ValueError(
        #         f"发现未定义的标签类型: {invalid_labels}\n"
        #         f"请更新 params.py 中的 LABEL_MAPPING: {config.LABEL_MAPPING}"
        #     )
        # y = y.values.astype(np.int32)
        y = df[config.LABEL_COLUMN].values
        # ========== 关键修改4：特征验证 ==========
        # # 检查特征列是否存在
        # missing_features = [col for col in config.FEATURE_COLUMNS if col not in df.columns]
        # if missing_features:
        #     raise ValueError(f"CSV文件中缺失特征列: {missing_features}")

        # 提取特征数据
        X = df[config.FEATURE_COLUMNS].values.astype(np.float32)

        # ========== 数据预处理 ==========
        # 标准化特征
        X = self.scaler.fit_transform(X)

        # 转换为one-hot编码
        y = to_categorical(y, num_classes=config.NUM_CLASSES)

        # ========== 数据分割 ==========
        # 分层抽样分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # 调试输出（可选）
        print("\n=== 数据加载验证 ===")
        print(f"特征矩阵形状: {X_train.shape} -> {X_test.shape}")
        print(f"标签分布示例: {np.unique(np.argmax(y_train, axis=1), return_counts=True)}")

        return X_train, X_test, y_train, y_test