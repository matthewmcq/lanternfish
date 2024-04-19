import tensorflow as tf
import numpy as np

def train(model, y_train, y_true, epochs=10, batch_size=1):

    loss_fn = WaveletLoss(model, l1_reg=5e-8, l2_reg=5e-9)

    optimizer = tf.keras.optimizers.Adam()
    # loss_fn = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanSquaredError()]

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    
    # train the model
    model.fit(x=y_train, y=y_true,  epochs=epochs, batch_size =batch_size, shuffle=True, validation_split=0.2)
    return model


class WaveletLoss(tf.keras.losses.Loss):
    def __init__(self, model, wavelet_level=4, lambda_vec=[40, 2.5, 0.3, 0.2], lambda_11=1, lambda_12=0.25, name='wavelet_loss',   l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.wavelet_level = wavelet_level
        self.lambda_vec = lambda_vec
        self.lambda_11 = lambda_11
        self.lambda_12 = lambda_12
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

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

        # Add regularization losses
        if self.l1_reg > 0:
            l1_loss = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in self.model.trainable_variables])
            loss += self.l1_reg * l1_loss

        if self.l2_reg > 0:
            l2_loss = tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_variables])
            loss += self.l2_reg * l2_loss

        return loss