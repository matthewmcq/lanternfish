import tensorflow as tf
import numpy as np
import soundfile as sf

def train(model, model_config, loss, train, val):
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
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
    BATCH_PARAMS = (model_config['wavelet_depth'], model_config['batch_size'], 5, 5)


    y_train, y_true, shape = batch_training_data(*BATCH_PARAMS)

    for i in range(10):
        prediction = model.predict(tf.expand_dims(y_train[i], axis=0))[0]
        true = np.transpose(y_true[i], (1,0))
        a3, d3, d2 = true
        sum_true = a3 + d3 + d2

        sum_pred = tf.squeeze(prediction, axis=-1)

        train = np.transpose(y_train[i], (1,0))
        a1, d1, d0 = train
        sum_train = a1 + d1 + d0

        sf.write(f'train_{i}.wav', sum_train, 22050)
        sf.write(f'true_{i}.wav', sum_true, 22050)
        sf.write(f'pred_{i}.wav', sum_pred, 22050)

    return model

@tf.keras.utils.register_keras_serializable()
class WaveletLoss(tf.keras.losses.Loss):
    def __init__(self, wavelet_level=4, lambda_vec=[10, 1000, 1000], lambda_11=1, lambda_12=0.25, name='wavelet_loss',   l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.wavelet_level = wavelet_level
        self.lambda_vec = lambda_vec
        self.lambda_11 = lambda_11
        self.lambda_12 = lambda_12
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    # # @tf.function
    def call(self, y_true, y_pred):

        # Sum the audios along the wavelet_filter dimension for each example in the batch
        summed_true = tf.math.reduce_sum(y_true, axis=-1)
        summed_pred = tf.math.reduce_sum(y_pred, axis=-1)

        # Calculate the mean squared error between the summed audios for each example in the batch
        mse = tf.math.reduce_mean(tf.math.square(summed_true - summed_pred))

        # Take the mean of the MSE across the batch
        return mse


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
