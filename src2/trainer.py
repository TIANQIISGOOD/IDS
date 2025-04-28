from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import tensorflow as tf
from params import config

class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.lr_scheduler = LearningRateScheduler(
            lambda epoch: config.INIT_LR * (0.5 ** (epoch // 10))
        )
        self.early_stop = EarlyStopping(
            patience=10,
            restore_best_weights=True
        )

    def train(self, X_train, y_train, X_val, y_val):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(config.INIT_LR),
            loss=self.model.build_custom_loss(),
            metrics=['accuracy',
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.AUC(name='auc')]
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=config.BATCH_SIZE,
            class_weight=config.CLASS_WEIGHTS,
            callbacks=[self.lr_scheduler, self.early_stop]
        )
        return history