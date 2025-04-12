import tensorflow as tf
from tensorflow.keras import layers, Model
from configs.params import config


class SENet(layers.Layer):
    def __init__(self, reduction_ratio=16):
        super(SENet, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.squeeze = layers.GlobalAveragePooling1D()
        channels = input_shape[-1]
        self.excitation = tf.keras.Sequential([
            layers.Dense(channels // self.reduction_ratio, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])

    def call(self, inputs):
        # Squeeze
        squeeze = self.squeeze(inputs)
        # Excitation
        excitation = self.excitation(squeeze)
        # Reshape为与输入相同的维度
        excitation = tf.expand_dims(excitation, axis=1)
        # Scale
        return inputs * excitation


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.query_dense = layers.Dense(d_model)
        self.key_dense = layers.Dense(d_model)
        self.value_dense = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # 线性变换
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # 分割头
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 缩放点积注意力
        scaled_attention_logits = tf.matmul(query, key, transpose_b=True)
        scaled_attention_logits = scaled_attention_logits / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # 注意力输出
        output = tf.matmul(attention_weights, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(output)
        return output


class BiLSTM_CNN(Model):
    def __init__(self, model_config=None):
        super(BiLSTM_CNN, self).__init__()

        # 使用默认配置或传入的配置
        self.config = {
            'cnn_filters': 64,
            'cnn_kernel_sizes': [5, 7, 9],
            'bilstm_units': [128, 64],
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.2,
            'spatial_attention_ratio': 16,  # SE模块的压缩比
            'temporal_attention_heads': 8,  # Transformer的头数
            'dense_units': [256, 128],
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
        self.spatial_attention = SENet(reduction_ratio=self.config['spatial_attention_ratio'])
        self.temporal_attention = MultiHeadSelfAttention(
            d_model=self.config['bilstm_units'][0] * 2,
            num_heads=self.config['temporal_attention_heads']
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

        # SE注意力
        spatial_context = self.spatial_attention(fused_features)

        # BiLSTM处理
        lstm_features = spatial_context
        for layer in self.bilstm_layers:
            lstm_features = layer(lstm_features)

        # Transformer自注意力
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