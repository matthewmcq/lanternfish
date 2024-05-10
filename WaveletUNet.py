import tensorflow as tf
import numpy as np


@tf.keras.utils.register_keras_serializable()
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

    # Create an instance of WaveletUNet with the extracted arguments
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_coeffs': self.num_coeffs,
            'wavelet_depth': self.wavelet_depth,
            'batch_size': self.batch_size,
            'channels': self.channels,
            'num_layers': self.num_layers,
            'num_init_filters': self.num_init_filters,
            'filter_size': self.filter_size,
            'merge_filter_size': self.merge_filter_size,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        return config

    def build(self, input_shape):
        # Create downsampling blocks
        self.downsampling_blocks = {}
        self.learnable_downsampling_blocks = {}
        self.P = {}
        self.U = {}
        for i in range(self.num_layers):
            block_name = f'{i+1}'
            num_filters = self.num_init_filters + (self.num_init_filters * i)
            self.downsampling_blocks[block_name] = DownsamplingLayer(num_filters, self.filter_size, name=block_name, l1_reg=self.l1_reg, l2_reg=self.l2_reg)
            # self.learnable_downsampling_blocks[block_name] = LearnableDownsamplingLayer(num_filters, self.filter_size, name=block_name, l1_reg=self.l1_reg, l2_reg=self.l2_reg)
            self.P[block_name] = tf.keras.layers.Conv1D(
                num_filters,
                3,
                activation=None,
                padding='same',
                name=f'P_{block_name}'
            )
            self.U[block_name] = tf.keras.layers.Conv1D(
                num_filters,
                3,
                activation=None,
                padding='same',
                name=f'U_{block_name}'
            )

        # Create bottle neck
        self.bottle_neck = tf.keras.layers.Conv1D(
            self.num_init_filters * (self.num_layers + 1),
            self.filter_size,
            activation='leaky_relu',
            padding='same',
            name='bottleneck_conv',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        )

        # Create upsampling blocks
        # self.upsampling_blocks = {}
        self.us_conv1d = {}
        self.even = {}
        self.odd = {}
        for i in range(self.num_layers):
            block_name = f'{self.num_layers - i}'
            num_filters = self.num_init_filters + (self.num_init_filters * (self.num_layers - i - 1))
            # out_channels = num_filters // 2

            # self.upsampling_blocks[block_name] = LearnableUpsamplingLayer(num_filters, self.merge_filter_size, name=block_name, l1_reg=self.l1_reg, l2_reg=self.l2_reg)

            self.us_conv1d[block_name] = tf.keras.layers.Conv1D(
                num_filters,
                self.merge_filter_size,
                activation='leaky_relu',
                padding='same',
                name=f'us_conv1d_{block_name}',
                trainable=True
            )
            self.even[block_name] = tf.keras.layers.Conv1D(
                num_filters,
                1,
                activation=None,
                padding='same',
                name=f'even_{block_name}',
                trainable=True
            )
            self.odd[block_name] = tf.keras.layers.Conv1D(
                num_filters,
                1,
                activation=None,
                padding='same',
                name=f'odd_{block_name}',
                trainable=True
            )


        self.output_conv3 = tf.keras.layers.Conv1D(
            1,
            1,
            activation='tanh',
            padding='same',
            name='output_conv3',
            # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            # activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        )
        super().build(input_shape)


    def call(self, inputs, is_training=True):


        current_layer = inputs

        full_mix = tf.math.reduce_sum(current_layer, axis=-1)

        enc_outputs = list()

        # Downsampling path
        for i in range(self.num_layers):

            block_name = f'{i+1}'

            current_layer = self.downsampling_blocks[block_name](current_layer)


            # Save for skip connections
            enc_outputs.append(current_layer)

            # Decimation step
            x_even, x_odd = current_layer[:, ::2, :], current_layer[:, 1::2, :]
            # print(f"shape of even: {x_even.shape}")
            # print(f"shape of odd: {x_odd.shape}")
            d = x_odd - self.P[block_name](x_even)

            c = x_even + self.U[block_name](d)

            A = 2**(1/2)

            c = c * A
            d = d * 1/A

            current_layer = tf.concat([c, d], axis=-1)


        # Bottle neck
        current_layer = self.bottle_neck(current_layer)

        # Upsampling path
        for i in range(self.num_layers):

            block_name = f'{self.num_layers - i}'

            x_even, x_odd = current_layer[:, :, :-current_layer.shape[-1]//2], current_layer[:, :, -current_layer.shape[-1]//2:]
            x_even = self.even[block_name](x_even)
            x_odd = self.odd[block_name](x_odd)
            # print(f"shape of even: {x_even.shape}")
            # print(f"shape of odd: {x_odd.shape}")

            A = 2**(1/2)
            x_odd *= A
            x_even *= 1/A

            # print(f"shape of u_setp: {u_step.shape}")
            c = x_even - self.U[block_name](x_odd)

            d = x_odd + self.P[block_name](c)

            # print(f"shape of d: {d.shape}")

            output = tf.concat([c, d], axis=1)
            # print(f"shape of output: {output.shape}")

            indices = []
            num_entries = x_even.shape[1]
            num_outputs = 2 * num_entries

            for idx in range(num_outputs):
                if idx % 2 == 0:
                    indices.append(idx // 2)
                else:
                    indices.append(num_entries + idx // 2)

            current_layer = tf.gather(output, indices, axis = 1)

            # Get skip connection
            skip_conn = enc_outputs[-i-1]

            # Pad if necessary
            desired_shape = skip_conn.shape

            ### NEW CROPPING METHOD -- crop current_layer to match skip_conn
            if current_layer.shape[1] != desired_shape[1]:
                if current_layer.shape[1] != desired_shape[1]:
                    diff = desired_shape[1] - current_layer.shape[1]
                    if diff >0:
                        pad_start = diff // 2
                        pad_end = diff - pad_start
                        current_layer = tf.pad(current_layer, [[0, 0], [pad_start, pad_end], [0,0]], mode='SYMMETRIC')
                    else:
                        diff = -diff
                        crop_start = diff // 2
                        current_layer = tf.slice(current_layer, [0, crop_start, 0], [-1, desired_shape[1], -1])


            # Concatenate with skip connection
            current_layer = tf.keras.layers.Concatenate()([current_layer, skip_conn])

            conv1d = self.us_conv1d[block_name]
            current_layer = conv1d(current_layer)



        desired_shape = full_mix.shape

        if current_layer.shape[1] != desired_shape[1]:
            diff = desired_shape[1] - current_layer.shape[1]
            if diff >0:
                pad_start = diff // 2
                pad_end = diff - pad_start
                current_layer = tf.pad(current_layer, [[0, 0], [pad_start, pad_end], [0,0]], mode='SYMMETRIC')
            else:
                diff = -diff
                crop_start = diff // 2
                current_layer = tf.slice(current_layer, [0, crop_start, 0], [-1, desired_shape[1], -1])


        current_layer = tf.keras.layers.Concatenate()([tf.expand_dims(full_mix, axis=-1), current_layer])

        current_layer = self.output_conv3(current_layer)

        return current_layer

@tf.keras.utils.register_keras_serializable()
class DownsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv1D(
            self.num_filters,
            self.filter_size,
            activation='leaky_relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=f'downsampling_conv_{self.num_filters}'
        )
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        return x


    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.num_filters,
            'filter_size': self.filter_size,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        return config

    def from_config(cls, config):
        return cls(**config)



