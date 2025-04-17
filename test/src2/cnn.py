import tensorflow as tf
from tensorflow.keras import layers
from params import config

def CNN():
    # 单层CNN基准模型
    model = tf.keras.Sequential([
        layers.Reshape((config.FEATURE_DIM, 1), input_shape=(config.FEATURE_DIM,)),
        layers.Conv1D(32, 3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.3),
        layers.Dense(config.NUM_CLASSES, activation='softmax')  # 8分类
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

def train_cnn(X_train, y_train, X_val=None, y_val=None):
    model = CNN()
    validation_data = (X_val, y_val) if X_val is not None else None
    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=validation_data,
        verbose=1
    )
    return model, history