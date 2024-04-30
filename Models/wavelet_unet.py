import tensorflow as tf
import numpy as np
import librosa
import sys

class WaveletUNet(tf.keras.Model):

    def __init__(self, num_coeffs, wavelet_depth, batch_size, channels, num_layers, num_init_filters, filter_size, merge_filter_size, l1_reg, l2_reg, **kwargs):
        super().__init__(**kwargs)
        self.num_coeffs = num_coeffs
        self.wavelet_depth = wavelet_depth + 1
        self.batch_size = batch_size
        self.channels = channels
        self.num_layers = num_layers
        self.num_init_filters = num_init_filters
        self.filter_size = filter_size
        self.merge_filter_size = merge_filter_size
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self.input_shape = (self.batch_size, self.num_coeffs, self.wavelet_depth)

    def get_config(self):
        return {
            'num_coeffs': self.num_coeffs,
            'wavelet_depth': self.wavelet_depth - 1,  # Subtract 1 to match the constructor argument
            'batch_size': self.batch_size,
            'channels': self.channels,
            'num_layers': self.num_layers,
            'num_init_filters': self.num_init_filters,
            'filter_size': self.filter_size,
            'merge_filter_size': self.merge_filter_size,  # Added this line to match the constructor argument
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        }

    @classmethod
    def from_config(cls, config):
        # Extract the necessary arguments from the config dictionary
        num_coeffs = config.pop('num_coeffs')
        wavelet_depth = config.pop('wavelet_depth')
        batch_size = config.pop('batch_size')
        channels = config.pop('channels')
        num_layers = config.pop('num_layers')
        num_init_filters = config.pop('num_init_filters')
        filter_size = config.pop('filter_size')
        merge_filter_size = config.pop('merge_filter_size')
        l1_reg = config.pop('l1_reg')
        l2_reg = config.pop('l2_reg')

        # Create an instance of WaveletUNet with the extracted arguments
        return cls(
            num_coeffs=num_coeffs,
            wavelet_depth=wavelet_depth,
            batch_size=batch_size,
            channels=channels,
            num_layers=num_layers,
            num_init_filters=num_init_filters,
            filter_size=filter_size,
            merge_filter_size=merge_filter_size,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            **config  # Pass any remaining arguments to the constructor
        )

    def build(self, input_shape):
        # Create downsampling blocks
        self.downsampling_blocks = {}
        for i in range(self.num_layers):
            block_name = f'ds{i+1}'
            num_filters = self.num_init_filters + (self.num_init_filters * i)
            self.downsampling_blocks[block_name] = DownsamplingLayer(num_filters, self.filter_size, name=block_name, l1_reg=self.l1_reg, l2_reg=self.l2_reg)

        # Create batch normalization layers for downsampling blocks
        self.DS_batch_norm = {}
        for i in range(self.num_layers):
            block_name = f'ds{i+1}'
            self.DS_batch_norm[block_name] = tf.keras.layers.BatchNormalization(axis=1, name=f'{block_name}_batch_norm')

        # Create bottle neck
        self.bottle_neck = tf.keras.layers.Conv2D(
            self.num_init_filters * self.num_layers,
            (self.filter_size, 1),
            activation='leaky_relu',
            padding='same',
            data_format='channels_last',
            name='output_conv1',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        )
        
        # Create upsampling blocks
        self.upsampling_blocks = {}
        self.us_conv1d = {}
        for i in range(self.num_layers):
            block_name = f'us{self.num_layers - i}'
            num_filters = self.num_init_filters + (self.num_init_filters * (self.num_layers - i - 1))
            # out_channels = num_filters // 2
            
            self.upsampling_blocks[block_name] = UpsamplingLayer(num_filters, self.merge_filter_size, name=block_name, l1_reg=self.l1_reg, l2_reg=self.l2_reg)

            self.us_conv1d[block_name] = tf.keras.layers.Conv2D(
                num_filters,
                (self.merge_filter_size, 1),
                activation='leaky_relu',
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=f'{block_name}_conv1d'
            )

        # Create batch normalization layers for upsampling blocks
        self.US_batch_norm = {}
        for i in range(self.num_layers):
            block_name = f'us{i+1}'
            self.US_batch_norm[block_name] = tf.keras.layers.BatchNormalization(axis=1, name=f'{block_name}_batch_norm')


        self.output_conv1 = tf.keras.layers.Conv2D(
            1,
            (self.merge_filter_size, 1),
            activation='leaky_relu',
            padding='same',
            data_format='channels_last',
            name='output_conv1',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        )

        self.output_conv2 = tf.keras.layers.Conv2D(
            1,
            (1, 1),
            activation='tanh',
            padding='same',
            data_format='channels_last',
            name='output_conv2',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        )
        super().build(input_shape)


    def call(self, inputs, is_training=True):
        
        current_layer = inputs 

        enc_outputs = list()

        # Downsampling path
        for i in range(self.num_layers):
            # print(f"DS {i} Layer input Shape: {current_layer.shape}")
            # Apply downsampling block
            block_name = f'ds{i+1}'
            block = self.downsampling_blocks[block_name]
            # print(f"shape before DS block {i}: { current_layer.shape} ")
            current_layer = block(current_layer)
            # print(f"shape after DS block {i}: { current_layer.shape} ")
            # Apply batch normalization
            if is_training:
                batch_norm = self.DS_batch_norm[block_name]
                current_layer = batch_norm(current_layer)

            # Save for skip connections
            enc_outputs.append(current_layer)
            
            # Decimation step
            current_layer = current_layer[:, ::2, :, :]
            # print(f"shape after DS {i} decimation: { current_layer.shape} ")
            # print(f"DS {i} Layer output Shape: {current_layer.shape}")
            

        # Bottle neck
        # print(f"Bottle Neck Layer input Shape: {current_layer.shape}")
        current_layer = self.bottle_neck(current_layer)
        # print(f"Bottle Neck Layer output Shape: {current_layer.shape}")
        # Upsampling path
        for i in range(self.num_layers):
            
            block_name = f'us{self.num_layers - i}'
            block = self.upsampling_blocks[block_name]
            #  current_layer = tf.image.resize_bilinear(current_layer, [1, current_layer.get_shape().as_list()[2]*2])
            # print(f"shape before US block {i}: {current_layer.shape}")
            current_layer = block(current_layer)
            # print(f"shape after US block {i}: {current_layer.shape}")
            # Get skip connection
            skip_conn = enc_outputs[-i-1]
            
            # Pad if necessary
            desired_shape = skip_conn.shape

            ### NEW CROPPING METHOD -- crop current_layer to match skip_conn
            if current_layer.shape[1] != desired_shape[1]:
                diff = current_layer.shape[1] - desired_shape[1]
                crop_start = diff // 2
                current_layer = tf.slice(current_layer, [0, crop_start, 0], [-1, desired_shape[1], -1])

            
            # Concatenate with skip connection
            current_layer = tf.keras.layers.Concatenate()([skip_conn, current_layer])
            # print(f"shape after US concat {i}: {current_layer.shape}")
            conv1d = self.us_conv1d[block_name]
            current_layer = conv1d(current_layer)
            # print(f"US {i} concat output Shape: {current_layer.shape}")

            # Apply batch normalization
            if is_training:
                batch_norm = self.US_batch_norm[block_name]
                current_layer = batch_norm(current_layer)

        

        # print(f"shape before out conv 1: {current_layer.shape}")
        # concatenate with the input
        current_layer = self.output_conv1(current_layer)
        # print(f"shape before final concat: {current_layer.shape}")
        desired_shape = inputs.shape

        if current_layer.shape[1] != desired_shape[1]:
            diff = desired_shape[1] - current_layer.shape[1]
            pad_start = diff // 2
            pad_end = diff - pad_start
            current_layer = tf.pad(current_layer, [[0, 0], [pad_start, pad_end], [0, 0]], mode='SYMMETRIC')

        current_layer = tf.keras.layers.Concatenate()([inputs, current_layer])
        # print(f"shape after final concat: {current_layer.shape}")
        
        # Final convolution layer, tanh activation
        current_layer = self.output_conv2(current_layer)

        # print(f"shape after final conv: {current_layer.shape}")
        return current_layer
    
    
class DownsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            self.num_filters,
            (self.filter_size, 1),
            activation='leaky_relu',
            padding='same',
            strides=(1, 1),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='downsampling_conv'
        )
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        return x

class UpsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, strides=2, l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            self.num_filters,
            (self.filter_size, 1),
            activation='leaky_relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='upsampling_conv'
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.image.resize(
            inputs,
            size=(inputs.shape[1] * self.strides, inputs.shape[2]),
            method=tf.image.ResizeMethod.LANCZOS5,
            antialias=True
        )
        x = self.conv(x)
        return x
    

class LearnableUpsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, strides=2, l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        

    def build(self, input_shape):
        features = input_shape[1]
        self.interp_weights = self.add_weight(
            name='interp_weights',
            shape=[features, input_shape[-1]],
            initializer='glorot_uniform',
            trainable=True,
            regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            constraint=tf.keras.constraints.NonNeg()
        )
        super().build(input_shape)

    def call(self, inputs):
        weights_scaled = tf.nn.sigmoid(self.interp_weights)  # Constrain weights to [0, 1]
        counter_weights = 1.0 - weights_scaled
        # print(weights_scaled.shape)
        weights_scaled = tf.stack([tf.linalg.diag(weights_scaled), tf.linalg.diag(counter_weights)], axis=0)
        # print(weights_scaled.shape)
        # weights_scaled = tf.expand_dims(weights_scaled, axis=0)
        # print(weights_scaled.shape)
        inputs = tf.expand_dims(inputs, axis=1)  # Add channel dimension
        # print(weights_scaled.shape)
        weights_scaled = tf.nn.conv2d(inputs, weights_scaled, strides=[1, 1, 1, 1], padding='SAME')
        # print(weights_scaled.shape)
        weights_scaled = tf.transpose(weights_scaled, [2, 0, 1, 3])
        # print(weights_scaled.shape)
        inputs = tf.transpose(inputs, [2, 0, 1, 3])
        # print(weights_scaled.shape)
        num_entries = inputs.shape[0]
        weights_scaled = tf.concat([inputs, weights_scaled], axis=0)
        # print(weights_scaled.shape)
        indices = []
        num_outputs = 2 * num_entries
        for idx in range(num_outputs):
            if idx % 2 == 0:
                indices.append(idx // 2)
            else:
                indices.append(num_entries + idx // 2)
        # print(weights_scaled.shape)
        weights_scaled = tf.gather(weights_scaled, indices)
        weights_scaled = tf.transpose(weights_scaled, [1, 2, 0, 3])
        weights_scaled = tf.squeeze(weights_scaled, axis=1)  # Remove channel dimension
        # print(weights_scaled.shape)
        return weights_scaled

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.num_filters,
            'filter_size': self.filter_size,
            'out_channels': self.out_channels,
            'strides': self.strides,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        return config
    
    
    
