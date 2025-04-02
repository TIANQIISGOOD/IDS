import tensorflow as tf
from tensorflow.keras import layers, Model
from configs.params import config


class ImprovedGRU(Model):
    def __init__(self):
        super(ImprovedGRU, self).__init__()

        # 1. 层次化GRU结构
        self.gru_layers = [
            layers.GRU(128, return_sequences=True,
                       recurrent_dropout=0.2,
                       kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.GRU(64, return_sequences=True,
                       recurrent_dropout=0.2,
                       kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ]

        # 2. 注意力机制
        self.attention = layers.Dense(1, activation='tanh')

        # 3. 改进的分类头
        self.classifier = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        # GRU特征提取
        x = inputs
        for layer in self.gru_layers:
            x = layer(x)

        # 注意力加权
        attention_weights = self.attention(x)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        context_vector = tf.reduce_sum(x * attention_weights, axis=1)

        # 分类
        return self.classifier(context_vector)

    def call(self, inputs):
        # GRU特征提取
        x = inputs
        for layer in self.gru_layers:
            x = layer(x)

        # 注意力加权
        attention_weights = self.attention(x)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        context_vector = tf.reduce_sum(x * attention_weights, axis=1)

        # 分类
        return self.classifier(context_vector)


def train_improved_gru(X_train, y_train, X_val, y_val):
    # 1. 模型实例化
    model = ImprovedGRU()

    # 2. 优化器配置
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        clipnorm=1.0  # 梯度裁剪
    )

    # 3. 早停策略
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # 4. 学习率调度
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # 5. 编译模型
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    # 6. 训练配置
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        class_weight={0: 1, 1: 5}  # 处理类别不平衡
    )

    return model, history