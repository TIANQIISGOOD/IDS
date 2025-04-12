import tensorflow as tf
from tensorflow.keras import layers, Model
from configs.params import config


class SpatialOnlyModel(Model):
    """仅保留空间特征提取模块（多尺度CNN + 空间注意力 + 分类头）"""

    def __init__(self):
        super(SpatialOnlyModel, self).__init__()

        # 多尺度CNN特征提取
        self.multi_scale_convs = []
        for kernel_size in [3, 5, 7]:
            conv_block = tf.keras.Sequential([
                layers.Conv1D(64, kernel_size=kernel_size, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2, padding='same'),
                layers.Dropout(0.2)
            ])
            self.multi_scale_convs.append(conv_block)

        # 特征融合层
        self.feature_fusion = layers.Concatenate()
        self.fusion_conv = layers.Conv1D(128, kernel_size=1, padding='same')

        # 空间注意力机制
        self.spatial_attention = layers.Dense(1, activation='tanh')

        # 分类头
        self.classifier = tf.keras.Sequential([
            layers.Dense(256, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        # 多尺度CNN特征提取
        conv_features = []
        for conv_block in self.multi_scale_convs:
            conv_features.append(conv_block(inputs))

        # 特征融合
        fused_features = self.feature_fusion(conv_features)
        fused_features = self.fusion_conv(fused_features)

        # 空间注意力
        attention_weights = self.spatial_attention(fused_features)
        attended_features = fused_features * attention_weights

        # 全局池化
        global_features = tf.reduce_mean(attended_features, axis=1)

        # 分类
        return self.classifier(global_features)

    def build_custom_loss(self):
        """自定义损失函数"""

        def custom_loss(y_true, y_pred):
            # 二元交叉熵
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
            return bce

        return custom_loss


class TemporalOnlyModel(Model):
    """仅保留时间特征提取模块（多层次BiLSTM + 时间注意力 + 分类头）"""

    def __init__(self):
        super(TemporalOnlyModel, self).__init__()

        # 多层次BiLSTM
        self.bilstm_layers = [
            layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                             recurrent_dropout=0.2)),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                                             recurrent_dropout=0.2))
        ]

        # 时间注意力机制
        self.temporal_attention = layers.Dense(1, activation='tanh')

        # 分类头
        self.classifier = tf.keras.Sequential([
            layers.Dense(256, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        # BiLSTM处理
        lstm_features = inputs
        for layer in self.bilstm_layers:
            lstm_features = layer(lstm_features)

        # 时间注意力
        attention_weights = self.temporal_attention(lstm_features)
        attended_features = lstm_features * attention_weights

        # 全局池化
        global_features = tf.reduce_mean(attended_features, axis=1)

        # 分类
        return self.classifier(global_features)

    def build_custom_loss(self):
        """自定义损失函数"""

        def custom_loss(y_true, y_pred):
            # 二元交叉熵
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
            return bce

        return custom_loss