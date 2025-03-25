import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, Model
from configs.params import config

class BiLSTM_CNN(Model):
    def __init__(self):
        super(BiLSTM_CNN, self).__init__()

        # CNN模块
        self.cnn = tf.keras.Sequential([
            # 输入形状应为 (batch_size, FEATURE_DIM)
            layers.Reshape((config.FEATURE_DIM, 1)),  # 转换为 (None, 102, 1)
            layers.Conv1D(
                filters=config.CNN_FILTERS,
                kernel_size=3,
                activation='relu',
                input_shape=(config.FEATURE_DIM, 1)  # 显式定义输入形状
            ),
            layers.MaxPooling1D(2),
            layers.Dropout(config.DROPOUT_RATE)
        ])

        # BiLSTM模块
        self.bilstm = tf.keras.Sequential([
            layers.Bidirectional(layers.LSTM(
                units=config.LSTM_UNITS,
                return_sequences=True
            )),
            layers.Bidirectional(layers.LSTM(
                units=config.LSTM_UNITS // 2
            ))
        ])

        # 分类头
        self.classifier = tf.keras.Sequential([
            layers.Dense(32, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(config.REGULARIZATION)),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        x = self.cnn(inputs)
        x = self.bilstm(x)
        return self.classifier(x)

    def build_custom_loss(self):
        """自定义损失函数"""

        def temporal_smoothness(y_true, y_pred):
            # 时序平滑正则项
            delta = tf.abs(y_pred[1:] - y_pred[:-1])
            return config.REGULARIZATION * tf.reduce_mean(delta)

        def total_loss(y_true, y_pred):
            bce = tf.keras.losses.BinaryCrossentropy()(
                y_true, y_pred,
                sample_weight=tf.where(y_true == 1, 5.0, 1.0)
            )
            return bce + temporal_smoothness(y_true, y_pred)

        return total_loss

