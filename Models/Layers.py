import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class UpsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, strides=2, l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
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
            name='upsampling_conv'
        )
        self.convtranspose = tf.keras.layers.Conv1DTranspose(
            self.num_filters,
            self.filter_size,
            strides=self.strides,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='upsampling_conv_reshape'
        )
        # self.lanczos = LanczosResizeLayer(scale_factor=self.strides, kernel_size=self.filter_size)
        super().build(input_shape)

    def call(self, inputs):
        # x = tf.image.resize(
        #     inputs,
        #     size=(inputs.shape[1] * self.strides, inputs.shape[2]),
        #     method=tf.image.ResizeMethod.LANCZOS5,
        #     antialias=True
        # )
        # x = self.lanczos(inputs)
        # print(f"shape after lanczos: {x.shape}")
        x = self.convtranspose(inputs)
        # print(f"shape after resize: {x.shape}")
        x = self.conv(x)
        return x

@tf.keras.utils.register_keras_serializable()
class LearnableUpsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, strides=2, l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg


    def build(self, input_shape):
        channels = input_shape[-1]
        wavelet_depth = input_shape[-2]
        self.interp_weights = self.add_weight(
            name='interp_weights',
            shape=[channels],
            initializer='glorot_uniform',
            trainable=True,
            regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
        )
        self.conv = tf.keras.layers.Conv1D(
            self.num_filters,
            self.filter_size,
            activation='leaky_relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='upsampling_conv'
        )
        super().build(input_shape)

    def call(self, inputs):
        weights_scaled = tf.nn.sigmoid(self.interp_weights)  # Constrain weights to [0, 1]
        counter_weights = 1.0 - weights_scaled
        # print("weights scaled shape", weights_scaled.shape)
        inputs = tf.expand_dims(inputs, axis=1) #

        # print("inputs shape", inputs.shape)
        weights_scaled = tf.concat([tf.expand_dims(tf.linalg.diag(weights_scaled), axis=0), tf.expand_dims(tf.linalg.diag(counter_weights), axis=0)], axis=0)
        # print("weights scaled shape", weights_scaled.shape)
        weights_scaled = tf.expand_dims(weights_scaled, axis=0) # (1,2,C,C)

        # print("weights scaled shape", weights_scaled.shape)
        # print(weights_scaled.shape)
        weights_scaled = tf.nn.conv2d(inputs, weights_scaled, strides=[1, 1, 1, 1], padding='SAME') #
        # print("weights scaled after conv", weights_scaled.shape)
        weights_scaled = tf.transpose(weights_scaled, [2,0,1,3])
        # print("weights scaled after transpose 1 ", weights_scaled.shape)

        inputs = tf.transpose(inputs, [2, 0, 1, 3])
        # print("inputs after transpose 1 ", inputs.shape)
        num_entries = inputs.shape[0]
        weights_scaled = tf.concat([inputs, weights_scaled], axis=0)
        # print("weights scaled after concat", weights_scaled.shape) # (2048, 8, 180, 4)
        indices = []
        num_outputs = 2 * num_entries
        for idx in range(num_outputs):
            if idx % 2 == 0:
                indices.append(idx // 2)
            else:
                indices.append(num_entries + idx // 2)

        weights_scaled = tf.gather(weights_scaled, indices)
        # print("weights scaled after gather", weights_scaled.shape)
        weights_scaled = tf.transpose(weights_scaled, [1, 2, 0, 3])
        # print("weights scaled after transpose 3", weights_scaled.shape)
        weights_scaled = tf.squeeze(weights_scaled, axis=1)
        # print("weights scaled after squeeze", weights_scaled.shape)
        weights_scaled = self.conv(weights_scaled)
        # print("weights scaled after conv", weights_scaled.shape)
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

# @tf.function(jit_compile = True)
@tf.keras.utils.register_keras_serializable()
class LanczosResizeLayer(tf.keras.layers.Layer):
    def __init__(self, scale_factor=2, kernel_size=5, **kwargs):
        super(LanczosResizeLayer, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.kernel = self.create_lanczos_kernel(self.kernel_size, self.scale_factor)
        self.kernel = tf.reshape(self.kernel, [(self.kernel_size-1),1, 1, 1])
        self.kernel = tf.tile(self.kernel, [1, 1, self.channels, self.channels])
        super(LanczosResizeLayer, self).build(input_shape)

    def call(self, inputs):
        batch_size, height, width, _ = inputs.shape
        if batch_size == None:
            batch_size, height, width, _ = tf.unstack(tf.shape(inputs))
        new_height = height * self.scale_factor
        new_width = width #* self.scale_factor

        # Create the Lanczos kernel


        # Expand the Lanczos kernel to match the number of channels


        # Perform the Lanczos upsampling convolution
        output = tf.nn.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape=[batch_size, new_height, new_width, self.channels],
            strides=[1, self.scale_factor, 1, 1],
            padding='SAME'
        )

        return output

    def create_lanczos_kernel(self, kernel_size, scale_factor):
        # Create the 1D Lanczos kernel
        a = kernel_size // scale_factor
        x = tf.range(-a + 1, a + 1, dtype=tf.float32)
        kernel_1d = self.sinc(x / a) * self.sinc(x)

        # Normalize the kernel
        kernel_1d /= tf.reduce_sum(kernel_1d)

        return kernel_1d

    @staticmethod
    def sinc(x):
        # x = tf.where(tf.equal(x, 0), tf.ones_like(x), x)
        y= tf.sin(np.pi * x) / (np.pi * x)
        y = tf.where(tf.math.is_nan(y), tf.ones_like(y), y)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            'scale_factor': self.scale_factor,
            'kernel_size': self.kernel_size
        })
        return config



