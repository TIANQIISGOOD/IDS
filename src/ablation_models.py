import tensorflow as tf
from tensorflow.keras import layers, Model
from .model import TemporalAttention,SpatialAttention


class SpatialOnlyModel(Model):
    """仅保留空间特征的模型（CNN + SENet）"""

    def __init__(self, config=None):
        super(SpatialOnlyModel, self).__init__()

        # 使用默认配置或传入的配置
        self.config = {
            'cnn_filters': 64,  # 'cnn_filters': n/2
            'cnn_kernel_sizes': [5, 7, 9],
            'bilstm_units': [128, 64],  # [第一层units, 第二层units]
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.2,
            'spatial_attention_kernel': 9,  # 与最大匹配
            'dense_units': [256, 128],  # 'dense_units': [n*2, n],
            'l2_reg': 0.001,
            'classifier_dropout': 0.3
        }
        if config:
            self.config.update(config)

        # 多尺度CNN特征提取
        self.multi_scale_convs = []
        for kernel_size in self.config['cnn_kernel_sizes']:
            conv_block = tf.keras.Sequential([
                layers.Conv1D(
                    self.config['cnn_filters'],
                    kernel_size=kernel_size,
                    padding='same',
                    activation='relu'
                ),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2, padding='same'),
                layers.Dropout(self.config['dropout_rate'])
            ])
            self.multi_scale_convs.append(conv_block)

        # 特征融合层
        self.feature_fusion = layers.Concatenate()
        self.fusion_conv = layers.Conv1D(
            self.config['cnn_filters'] * 2,
            kernel_size=1,
            padding='same'
        )

        # 空间注意力
        self.spatial_attention = SpatialAttention(
            kernel_size=self.config['spatial_attention_kernel']
        )


        # 分类头
        self.classifier = self._build_classifier()

    def _build_classifier(self):
        classifier = tf.keras.Sequential()
        for units in self.config['dense_units']:
            classifier.add(layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.config['l2_reg'])
            ))
            classifier.add(layers.BatchNormalization())
            classifier.add(layers.Dropout(self.config['classifier_dropout']))
        classifier.add(layers.Dense(1, activation='sigmoid'))
        return classifier

    def call(self, inputs):
        # 多尺度CNN特征提取
        conv_features = []
        for conv_block in self.multi_scale_convs:
            conv_features.append(conv_block(inputs))

        # 特征融合
        fused_features = self.feature_fusion(conv_features)
        fused_features = self.fusion_conv(fused_features)

        # 空间注意力
        spatial_context = self.spatial_attention(fused_features)

        # 全局池化
        global_features = tf.reduce_mean(spatial_context, axis=1)

        # 分类
        return self.classifier(global_features)

    def build_custom_loss(self):
        def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

            pt_1 = tf.clip_by_value(pt_1, 1e-9, 1.0)
            pt_0 = tf.clip_by_value(pt_0, 1e-9, 1.0)

            return -tf.reduce_mean(
                alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1) +
                (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0)
            )

        def temporal_smoothness(y_true, y_pred):
            delta = tf.abs(y_pred[1:] - y_pred[:-1])
            return 0.005 * tf.reduce_mean(delta)

        def total_loss(y_true, y_pred):
            return focal_loss(y_true, y_pred) + temporal_smoothness(y_true, y_pred)

        return total_loss


class TemporalOnlyModel(Model):
    """仅保留时间特征的模型（BiLSTM + Transformer自注意力）"""

    def __init__(self, config=None):
        super(TemporalOnlyModel, self).__init__()

        # 使用默认配置或传入的配置
        self.config = {
            'cnn_filters': 64,  # 'cnn_filters': n/2
            'cnn_kernel_sizes': [5, 7, 9],
            'bilstm_units': [128, 64],  # [第一层units, 第二层units]
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.2,
            'spatial_attention_kernel': 9,  # 与最大匹配
            'dense_units': [256, 128],  # 'dense_units': [n*2, n],
            'l2_reg': 0.001,
            'classifier_dropout': 0.3
        }
        if config:
            self.config.update(config)

        # 层次化BiLSTM
        self.bilstm_layers = []
        for units in self.config['bilstm_units']:
            self.bilstm_layers.extend([
                layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=True,
                        recurrent_dropout=self.config['recurrent_dropout']
                    )
                ),
                layers.BatchNormalization()
            ])

        # 时间注意力
        self.temporal_attention = TemporalAttention(self.config['bilstm_units'][0] * 2)

        # 分类头
        self.classifier = self._build_classifier()

    def _build_classifier(self):
        classifier = tf.keras.Sequential()
        for units in self.config['dense_units']:
            classifier.add(layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.config['l2_reg'])
            ))
            classifier.add(layers.BatchNormalization())
            classifier.add(layers.Dropout(self.config['classifier_dropout']))
        classifier.add(layers.Dense(1, activation='sigmoid'))
        return classifier

    def call(self, inputs):
        # BiLSTM处理
        lstm_features = inputs
        for layer in self.bilstm_layers:
            lstm_features = layer(lstm_features)

        # 时间自注意力
        temporal_context = self.temporal_attention(lstm_features)

        # 全局池化
        global_features = tf.reduce_mean(temporal_context, axis=1)

        # 分类
        return self.classifier(global_features)

    def build_custom_loss(self):
        def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

            pt_1 = tf.clip_by_value(pt_1, 1e-9, 1.0)
            pt_0 = tf.clip_by_value(pt_0, 1e-9, 1.0)

            return -tf.reduce_mean(
                alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1) +
                (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0)
            )

        def temporal_smoothness(y_true, y_pred):
            delta = tf.abs(y_pred[1:] - y_pred[:-1])
            return 0.005 * tf.reduce_mean(delta)

        def total_loss(y_true, y_pred):
            return focal_loss(y_true, y_pred) + temporal_smoothness(y_true, y_pred)

        return total_loss