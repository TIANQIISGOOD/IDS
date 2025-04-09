import tensorflow as tf
from tensorflow.keras import layers, Model
from configs.params import config


class TemporalAttention(layers.Layer):
    def __init__(self, units):
        super(TemporalAttention, self).__init__()
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W(inputs)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = inputs * attention_weights
        return context_vector


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):  # 添加kernel_size参数
        super(SpatialAttention, self).__init__()
        self.conv1d = layers.Conv1D(1, kernel_size=kernel_size, padding='same')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=2, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=2, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=2)
        spatial_attention = tf.nn.sigmoid(self.conv1d(concat))
        return inputs * spatial_attention


class BiLSTM_CNN(Model):
    def __init__(self, model_config=None):
        super(BiLSTM_CNN, self).__init__()

        # 使用默认配置或传入的配置
        self.config = {
            'cnn_filters': 64,#'cnn_filters': n/2
            'cnn_kernel_sizes': [5, 7, 9],
            'bilstm_units': [128, 64],  # [第一层units, 第二层units]
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.2,
            'spatial_attention_kernel': 9,#与最大匹配
            'dense_units': [256, 128],#'dense_units': [n*2, n],
            'l2_reg': 0.001,
            'classifier_dropout': 0.3
        }
        if model_config:
            self.config.update(model_config)

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

        # 注意力机制
        self.temporal_attention = TemporalAttention(self.config['bilstm_units'][0] * 2)
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

        # 3. 空间注意力 (提前到BiLSTM之前)
        spatial_context = self.spatial_attention(fused_features)

        # BiLSTM处理
        lstm_features = spatial_context
        for layer in self.bilstm_layers:
            lstm_features = layer(lstm_features)

        # 时间注意力
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


