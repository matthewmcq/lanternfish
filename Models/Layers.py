import tensorflow as tf

class InterpolationLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, layer_number, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.layer_number = layer_number # for naming purposes

    def build(self, input_shape):
        '''
        :param input_shape: Input shape; should be [batch_size, 1, width, F] FOR STFT. NOT NECESSARILY FOR THIS
        dwt test shape: [batch_size = None, (1), 22 = (SR / 2) * time, ]

        basically if we use the logic from wave-u-net of expanding dims at axis 1, the input will then be:
        [batch_size, 1, num_features, wavelet_depth]
        '''
        features = input_shape[-1] # number of feature channels in input map
        self.weights = self.add_weight(
            shape=[features],
            initializer='glorot_uniform', # TODO: possibly change this later?
            trainable=True,
            name=f'interp_{self.layer_number}'
        )

        # Scale weights to [0, 1] and create diagonal matrix
        # Expected dims of weights_mat after expansion: [1, F, F]
        scaled_weights = tf.nn.sigmoid(self.weights)
        weights_mat = tf.expand_dims(tf.linalg.diag(scaled_weights), axis=0)

        # Get counter weights (representing DS layer on other side of skip connection) and create diagonal matrix
        # Expected dims of counter_weights_mat after expansion: [1, F, F]
        counter_weights = 1.0 - scaled_weights
        counter_weights_mat = tf.expand_dims(tf.linalg.diag(counter_weights), axis=0)

        # Concatenate the weights + counter weights along axis 0 to generate 2FxF weight matrix
        # and expand dims to include batch_size dimension in preparation for conv
        # Expected dims of conv_weights after expansion: [batch_size, wavelet_scales, F, F]
        #   where:
        #       batch_size = 1
        #       number of wavelet scales (original + counter) = 2
        #       F = number of feature channels / wavelet frequency bands
        #   note: batch_size=1 because we are only processing one input at a time. if we did more than one, we would be processing
        #   the same weight matrix repeatedly.
        conv_weights = tf.expand_dims(tf.concat(weights_mat, counter_weights_mat, axis=0), axis=0)

        # number of filter = 2F because that is the dimension of the concatenated weight matrix
        self.conv = tf.keras.layers.Conv2D(filters=2*features, kernel_size=(1, 1), use_bias=False, padding='same', kernel_initializer=conv_weights)

        super().build(input_shape)

    def call(self, input):
        conv_output = self.conv(input)

        # Transpose the conv output and the original input to have shape [width, batch_size, features, height]
        # in order to concatenate along the width axis
        #   note: in the context of our use of DWT, the width axis is the time-frequency axis, and the height axis
        #   is the wavelet depth axis
        conv_output = tf.transpose(conv_output, [3, 0, 1, 2])
        out = tf.transpose(input, [3, 0, 1, 2])

        # Concatenate the two along the width axis
        out = tf.concat([out, conv_output], axis=0)



############################################################################################################################################
###### NOTES #######
# current_layer shape: (32, 22, 182) at block us12
# current_layer shape: (32, 44, 182) at block us11
# current_layer shape: (32, 87, 182) at block us10
# current_layer shape: (32, 173, 182) at block us9
# current_layer shape: (32, 345, 182) at block us8
# current_layer shape: (32, 690, 182) at block us7
# current_layer shape: (32, 1379, 182) at block us6
# current_layer shape: (32, 2757, 182) at block us5
# current_layer shape: (32, 5513, 182) at block us4
# current_layer shape: (32, 11025, 182) at block us3
# current_layer shape: (32, 22050, 182) at block us2
# current_layer shape: (32, 44100, 182) at block us1

# In each of these steps, the second dimension is equal to 44100 / 2^i, where i is the index of the block (us1, us2, etc.) minus 1.

