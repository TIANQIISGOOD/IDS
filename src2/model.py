import tensorflow as tf
from tensorflow.keras import layers, Model
from params import config


class TemporalAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W(inputs)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = inputs * attention_weights
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv1D(1, kernel_size=kernel_size, padding='same')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_map = self.conv(concat)
        spatial_map = tf.nn.sigmoid(spatial_map)
        return inputs * spatial_map

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size
        })
        return config


class BiLSTM_CNN(Model):
    def __init__(self, config_dict=None, **kwargs):
        super(BiLSTM_CNN, self).__init__(**kwargs)

        # 使用默认配置或传入的配置
        self.model_config = {
            'cnn_filters': 64,
            'cnn_kernel_sizes': [5, 7, 9],
            'bilstm_units': [128, 64],
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.2,
            'spatial_attention_kernel': 9,
            'dense_units': [256, 128],
            'l2_reg': 0.001,
            'classifier_dropout': 0.3,
            'num_classes': config.NUM_CLASSES
        }
        if config_dict:
            self.model_config.update(config_dict)

        # 多尺度CNN特征提取
        self.multi_scale_convs = []
        for kernel_size in self.model_config['cnn_kernel_sizes']:
            conv_block = tf.keras.Sequential([
                layers.Conv1D(
                    self.model_config['cnn_filters'],
                    kernel_size=kernel_size,
                    padding='same',
                    activation='relu'
                ),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2, padding='same'),
                layers.Dropout(self.model_config['dropout_rate'])
            ])
            self.multi_scale_convs.append(conv_block)

        # 特征融合层
        self.feature_fusion = layers.Concatenate()
        self.fusion_conv = layers.Conv1D(
            self.model_config['cnn_filters'] * 2,
            kernel_size=1,
            padding='same'
        )

        # 双向LSTM层
        self.bilstm_layers = []
        for units in self.model_config['bilstm_units']:
            self.bilstm_layers.extend([
                layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=True,
                        recurrent_dropout=self.model_config['recurrent_dropout']
                    )
                ),
                layers.BatchNormalization()
            ])

        # 注意力机制
        self.temporal_attention = TemporalAttention(self.model_config['bilstm_units'][0] * 2)
        self.spatial_attention = SpatialAttention(
            kernel_size=self.model_config['spatial_attention_kernel']
        )

        # 分类头
        self.classifier_layers = []
        for units in self.model_config['dense_units']:
            self.classifier_layers.extend([
                layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(self.model_config['l2_reg'])
                ),
                layers.BatchNormalization(),
                layers.Dropout(self.model_config['classifier_dropout'])
            ])
        self.classifier_layers.append(
            layers.Dense(self.model_config['num_classes'], activation='softmax')
        )

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

        # 双重注意力机制
        temporal_context = self.temporal_attention(lstm_features)
        spatial_context = self.spatial_attention(temporal_context)

        # 全局池化
        global_features = tf.reduce_mean(spatial_context, axis=1)

        # 分类
        x = global_features
        for layer in self.classifier_layers:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "config_dict": self.model_config
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config_dict=config.get("config_dict", None))

    def build_custom_loss(self):
        def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
            epsilon = 1e-7
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            ce = -y_true * tf.math.log(y_pred)
            weight = tf.pow(1. - y_pred, gamma) * y_true
            fl = alpha * weight * ce
            return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

        def temporal_smoothness(y_true, y_pred):
            delta = tf.abs(y_pred[1:] - y_pred[:-1])
            return 0.005 * tf.reduce_mean(delta)

        def total_loss(y_true, y_pred):
            return categorical_focal_loss(y_true, y_pred) + temporal_smoothness(y_true, y_pred)

        return total_loss