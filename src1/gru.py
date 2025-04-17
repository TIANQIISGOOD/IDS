import tensorflow as tf
from tensorflow.keras import layers
from params import config


def GRU():
    # 单层GRU基准模型
    model = tf.keras.Sequential([
        layers.Reshape((config.FEATURE_DIM, 1), input_shape=(config.FEATURE_DIM,)),
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(config.NUM_CLASSES, activation='softmax')  # 8分类
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model


def train_improved_gru(X_train, y_train, X_val, y_val):
    model = GRU()

    # 基本的早停策略
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # 训练配置
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    return model, history