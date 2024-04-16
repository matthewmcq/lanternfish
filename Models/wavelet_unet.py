import tensorflow as tf
import numpy as np
import librosa
import sys


class WaveletUNet(tf.keras.Model):

    def __init__(self, model_config):
        super().__init__()
        self.num_coeffs = model_config['num_coeffs']
        self.wavelet_depth = model_config['wavelet_depth']
        self.batch_size = model_config['batch_size']
        self.channels = model_config['channels']
        self.num_layers = model_config['num_layers']
        self.num_init_filters = model_config['num_init_filters']
        self.filter_size = model_config['filter_size']

        self.input_shape = (self.wavelet_depth+1, self.num_coeffs, self.channels)

        # Create downsampling blocks TODO: Might make this a dict
        self.downsampling_blocks = [
            DownsamplingLayer(self.num_init_filters * (2 ** i), self.filter_size)
            for i in range(self.num_layers)
        ]

        # Create upsampling blocks
        self.upsampling_blocks = [
            UpsamplingLayer(self.num_init_filters * (2 ** (self.num_layers - i - 1)), self.filter_size)
            for i in range(self.num_layers)
        ]


        ## TODO: (IMPORTANT) MAKE 342 A PARAMETER, @rayhan this is probably why there was so much effort that went into the padding
        ## for the Wave-U-Net model
        self.last_crop = tf.keras.layers.Cropping1D((342, 342), name='last_crop')
                                                    # ^^^^^^THESE SHOULD NOT BE HARD CODED!!! 

        # Final convolution layer -- pretty sure tanh is the right activation function here bc wavelets can be negative
        self.output_conv = tf.keras.layers.Conv1D(self.channels, 1, activation='tanh', name='output_conv')


    def call(self, inputs):

        # right now inputs should be of shape (batch_size, num_coeffs, channels)
        
        current_layer = inputs[0] # need [0] because of the way the data is batched... TODO: fix this and investigate...

        # store outputs for skip connections        
        enc_outputs = list()

        # Downsampling path
        for block in self.downsampling_blocks:

            # current downsampling block
            current_layer = block(current_layer)

            # store output for skip connection
            enc_outputs.append(current_layer)

            # desimate along coefficients
            current_layer = current_layer[:, ::2, :]

        # Upsampling path
        for i, block in enumerate(self.upsampling_blocks):

            # current upsampling block
            current_layer = block(current_layer)

            # find associated skip connection
            skip_conn = enc_outputs[-i-1]

            # pad skip connection if necessary, need desired shape
            desired_shape = skip_conn.shape
            
            if current_layer.shape[1] != desired_shape[1]:
                ## pad smaller tensor with last value
                pad = current_layer.shape[1] - desired_shape[1]

                ## pad the skip connection -- we might think about LERP, but this should just end up getting fixed
                ## when we switch to a learned upsampling layer
                skip_conn = tf.pad(skip_conn, [[0, 0], [0, pad], [0, 0]], 'CONSTANT')

            # concatenate skip connection
            current_layer = tf.keras.layers.Concatenate()([current_layer, skip_conn])

        # Crop off the last 342 samples from both sides
        # currently goes from front and back, but we might want to just go from the back
        current_layer = self.last_crop(current_layer)

        # Final convolution layer, sigmoid activation
        output = self.output_conv(current_layer)
        return output
    

# TODO: Move to Train.py at some point -- maybe not rn for convenience, but definitely later
def train(model, train_data, epochs=10, batch_size=1):

    # I feel like these could go in a config file or kwargs
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.RootMeanSquaredError()]

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # print(f"Training data {train_data}")

    # batch the data
    train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # print(f"Training data after batching: {train_data}")

    # train the model
    model.fit(train_data, epochs=epochs)

    return model

### TODO: Might want to put these in a separate file (e.g. Layers.py) 
class DownsamplingLayer: ## TODO: add get_config if something fucks up when we get to loading (a la hw 5)

    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size

        self.conv = tf.keras.layers.Conv1D(self.num_filters, self.filter_size, activation='leaky_relu', padding='same',strides=1, name='downsampling_conv')

    def __call__(self, inputs):
        x = self.conv(inputs)
        return x
    

class UpsamplingLayer: ## TODO: add get_config if something fucks up when we get to loading (a la hw 5)

    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size

        self.conv = tf.keras.layers.Conv1D(self.num_filters, self.filter_size, activation='leaky_relu', padding='same',strides=1, name='upsampling_conv')
        self.up = tf.keras.layers.UpSampling1D(2, name='upsampling')


    def __call__(self, inputs):
        x = self.conv(inputs)
        x = self.up(x)
        return x