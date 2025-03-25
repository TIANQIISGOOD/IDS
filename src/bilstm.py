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

def train_bilstm(X_train, y_train):
    model = BiLSTM()
    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.3,
        verbose=0
    )
    return model, history