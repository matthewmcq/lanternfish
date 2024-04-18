import tensorflow as tf
import numpy as np
import librosa
import sys


class WaveletUNet(tf.keras.Model):

    def __init__(self, model_config):
        super().__init__()
        self.num_coeffs = model_config['num_coeffs']
        self.wavelet_depth = model_config['wavelet_depth'] + 1
        self.batch_size = model_config['batch_size']
        self.channels = model_config['channels']
        self.num_layers = model_config['num_layers']
        self.num_init_filters = model_config['num_init_filters']
        self.filter_size = model_config['filter_size']

        self.input_shape = (self.batch_size, self.num_coeffs, self.wavelet_depth)

    def build(self, input_shape):
        # Create downsampling blocks
        self.downsampling_blocks = {}
        for i in range(self.num_layers):
            block_name = f'ds{i+1}'
            num_filters = self.num_init_filters + (self.num_init_filters * i)
            self.downsampling_blocks[block_name] = DownsamplingLayer(num_filters, self.filter_size, name=block_name)


        self.bottle_neck = tf.keras.layers.Conv1D(self.num_init_filters + (self.num_init_filters * self.num_layers), self.filter_size, activation='leaky_relu', padding='same', strides=1, name='bottle_neck')
        # Create upsampling blocks
        self.upsampling_blocks = {}
        for i in range(self.num_layers):
            block_name = f'us{i+1}'
            num_filters = self.num_init_filters + (self.num_init_filters * (self.num_layers - i - 1))
            self.upsampling_blocks[block_name] = UpsamplingLayer(num_filters, self.filter_size, name=block_name)

        # Cropping layer
        self.last_crop = tf.keras.layers.Cropping1D(cropping=(956, 956), name='last_crop')

        # Final convolution layer
        self.output_conv = tf.keras.layers.Conv1D(self.wavelet_depth, 1, activation='tanh', name='output_conv')
        
        super().build(input_shape)


    def call(self, inputs):
        
        current_layer = inputs  # need [0] because of the way the data is batched... TODO: fix this and investigate...

        enc_outputs = list()

        # Downsampling path
        for i in range(self.num_layers):
            block_name = f'ds{i+1}'
            block = self.downsampling_blocks[block_name]
            current_layer = block(current_layer)
            enc_outputs.append(current_layer)
            current_layer = current_layer[:, ::2, :]

        # Bottle neck
        current_layer = self.bottle_neck(current_layer)

        # Upsampling path
        for i in range(self.num_layers):
            block_name = f'us{self.num_layers - i}'
            block = self.upsampling_blocks[block_name]
            current_layer = block(current_layer)

            skip_conn = enc_outputs[-i-1]
            desired_shape = skip_conn.shape
            
            if current_layer.shape[1] != desired_shape[1]:
                pad = current_layer.shape[1] - desired_shape[1]
                skip_conn = tf.pad(skip_conn, [[0, 0], [0, pad], [0, 0]], 'CONSTANT')

            current_layer = tf.keras.layers.Concatenate()([current_layer, skip_conn])

        # Crop off the last 342 samples from both sides
        current_layer = self.last_crop(current_layer)

        # Final convolution layer, sigmoid activation
        output = self.output_conv(current_layer)

        return output
    

### TODO: Figure out how to put these in Layers.py
class DownsamplingLayer(tf.keras.layers.Layer):

    def __init__(self, num_filters, filter_size, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv1D(
            self.num_filters,
            self.filter_size,
            activation='leaky_relu',
            padding='same',
            strides=1,
            name='downsampling_conv'
        )
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        return x

class UpsamplingLayer(tf.keras.layers.Layer): ## TODO: Implement Interpolation Layer

    def __init__(self, num_filters, filter_size, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides

    def build(self, input_shape):
        self.conv_transpose = tf.keras.layers.Conv1DTranspose(
            self.num_filters,
            self.filter_size,
            strides=self.strides,
            activation='leaky_relu',
            padding='same',
            name='upsampling_conv_transpose'
        )
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        return x