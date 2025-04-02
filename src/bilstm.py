import tensorflow as tf
from tensorflow.keras import layers
from configs.params import config

def BiLSTM():
    model = tf.keras.Sequential([
        layers.Reshape((config.FEATURE_DIM, 1), input_shape=(config.FEATURE_DIM,)),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(1, activation='sigmoid')
    ], name="BiLSTM")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

def train_bilstm(X_train, y_train, X_val=None, y_val=None):
    model = BiLSTM()
    validation_data = (X_val, y_val) if X_val is not None else None
    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=validation_data,
        verbose=1
    )
    return model, history