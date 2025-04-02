import tensorflow as tf
from tensorflow.keras import layers, Model
from configs.params import config


class TemporalAttention(layers.Layer):
    def __init__(self, units):
        super(TemporalAttention, self).__init__()
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        score = self.V(tf.nn.tanh(self.W(inputs)))  # (batch_size, time_steps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = inputs * attention_weights
        return context_vector


class SpatialAttention(layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1d = layers.Conv1D(1, kernel_size=7, padding='same')

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        avg_pool = tf.reduce_mean(inputs, axis=2, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=2, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=2)
        spatial_attention = tf.nn.sigmoid(self.conv1d(concat))
        return inputs * spatial_attention


class BiLSTM_CNN(Model):
    def __init__(self):
        super(BiLSTM_CNN, self).__init__()

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

        # 层次化BiLSTM
        self.bilstm_layers = [
            layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                             recurrent_dropout=0.2)),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                                             recurrent_dropout=0.2))
        ]

        # 注意力机制
        self.temporal_attention = TemporalAttention(128)
        self.spatial_attention = SpatialAttention()

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

        # BiLSTM处理
        lstm_features = fused_features
        for layer in self.bilstm_layers:
            lstm_features = layer(lstm_features)

        # 双重注意力
        temporal_context = self.temporal_attention(lstm_features)
        spatial_context = self.spatial_attention(temporal_context)

        # 全局池化
        global_features = tf.reduce_mean(spatial_context, axis=1)

        # 分类
        return self.classifier(global_features)

    def build_custom_loss(self):
        def focal_loss(y_true, y_pred):
            gamma = 2.0
            alpha = 0.25

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