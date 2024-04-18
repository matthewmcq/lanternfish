import tensorflow as tf
import numpy as np

def train(model, y_train, y_true, epochs=10, batch_size=1):

    loss_fn = WaveletLoss()

    optimizer = tf.keras.optimizers.Adam()
    # loss_fn = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    model.summary()

    # print the model summary
    

    # print(f"Training data {train_data}")
    # print(f"Training data shape: {train_data}")

    # batch the data
    # train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # print(f"Training data after batching: {train_data}")

    # train the model
    model.fit(x=y_train, y=y_true,  epochs=epochs, batch_size =batch_size, shuffle=True, validation_split=0.2)
    return model


class WaveletLoss(tf.keras.losses.Loss):
    def __init__(self, wavelet_level=4, lambda_vec=[40, 2.5, 0.3, 0.2], lambda_11=1, lambda_12=0.25, name='wavelet_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.wavelet_level = wavelet_level
        self.lambda_vec = lambda_vec
        self.lambda_11 = lambda_11
        self.lambda_12 = lambda_12

    # @tf.function
    def call(self, y_true, y_pred):
        # First index are approximation (midband) coefficients, second index are detail coefficients
        loss = self.lambda_11 * tf.keras.losses.mean_squared_error(y_true[:, :, 0], y_pred[:, :, 0])
        loss += self.lambda_12 * tf.keras.losses.mean_absolute_error(y_true[:, :, 1], y_pred[:, :, 1])
        loss *= self.lambda_vec[0]

        # For levels 2 through second to last, take MAE of detail coefficients times lambda
        for i in range(2, self.wavelet_level):
            loss += self.lambda_vec[i-1] * tf.keras.losses.mean_absolute_error(y_true[:, :, i], y_pred[:, :, i])

        # For the last level, take MSE of detail coefficients times lambda
        loss += self.lambda_vec[-1] * tf.keras.losses.mean_squared_error(y_true[:, :, -1], y_pred[:, :, -1])

        return loss