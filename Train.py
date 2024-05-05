import tensorflow as tf
import numpy as np

def train(model, model_config, loss, train, val):
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        start_from_epoch=0
    )

    
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_config['learning_rate'])
    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanSquaredError()]
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    model.fit(
        train,
        epochs=model_config['epochs'],
        validation_data=val,
        callbacks=[es]
        
    )

    return model

class WaveletLoss(tf.keras.losses.Loss):
    def __init__(self, wavelet_level=4, lambda_vec=[40, 2.5, 0.3, 0.2], lambda_11=1, lambda_12=0.25, name='wavelet_loss',   l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.wavelet_level = wavelet_level
        self.lambda_vec = lambda_vec
        self.lambda_11 = lambda_11
        self.lambda_12 = lambda_12
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        # self.model_config = model_config

    # @tf.function
    def call(self, y_true, y_pred):
        # First index are approximation (midband) coefficients, second index are detail coefficients
        # loss = self.lambda_11 * tf.keras.losses.mean_squared_error(y_true[:, :, 0, :], y_pred[:, :, 0, :])
        # loss += self.lambda_12 * tf.keras.losses.mean_squared_error(y_true[:, :, 1, :], y_pred[:, :, 1, :])
        # loss *= self.lambda_vec[0]

        # # For levels 2 through second to last, take MAE of detail coefficients times lambda
        # for i in range(2, self.wavelet_level):
        #     loss += self.lambda_vec[i-1] * tf.keras.losses.mean_squared_error(y_true[:, :, i, :], y_pred[:, :, i, :])

        # # For the last level, take MSE of detail coefficients times lambda
        # loss += self.lambda_vec[-1] * tf.keras.losses.mean_squared_error(y_true[:, :, -1, :], y_pred[:, :, -1, :])

        loss = tf.math.reduce_mean(tf.math.square((y_true[:, :, 0] - y_pred[:, :, 0])))        

        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'wavelet_level': self.wavelet_level,
            'lambda_vec': self.lambda_vec,
            'lambda_11': self.lambda_11,
            'lambda_12': self.lambda_12,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
class WaveletLoss2(tf.keras.losses.Loss):
    def __init__(self, wavelet_level=4, lambda_vec=[40, 2.5, 0.3, 0.2], lambda_11=1, lambda_12=0.25, name='wavelet_loss2',   l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.wavelet_level = wavelet_level
        self.lambda_vec = lambda_vec
        self.lambda_11 = lambda_11
        self.lambda_12 = lambda_12
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    # @tf.function
    def call(self, y_true, y_pred):
        # First index are approximation (midband) coefficients, second index are detail coefficients
        loss = self.lambda_11 * tf.math.reduce_mean(tf.math.square((y_true[:, :, 0, :] - y_pred[:, :, 0, :])))
        loss += self.lambda_12 * tf.math.reduce_mean(tf.math.square((y_true[:, :, 1, :] - y_pred[:, :, 1, :])))
        loss *= self.lambda_vec[0]

        # For levels 2 through second to last, take MAE of detail coefficients times lambda
        for i in range(2, self.wavelet_level):
            loss += self.lambda_vec[i-1] * tf.math.reduce_mean(tf.math.square((y_true[:, :, i, :] - y_pred[:, :, i, :])))

        # For the last level, take MSE of detail coefficients times lambda
        loss += self.lambda_vec[-1] * tf.math.reduce_mean(tf.math.square((y_true[:, :, -1, :] - y_pred[:, :, -1, :])))



        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'wavelet_level': self.wavelet_level,
            'lambda_vec': self.lambda_vec,
            'lambda_11': self.lambda_11,
            'lambda_12': self.lambda_12,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)