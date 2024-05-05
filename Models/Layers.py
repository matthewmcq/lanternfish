import tensorflow as tf

class InterpolationLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, layer_number, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.layer_number = layer_number # for naming purposes

    def build(self, input_shape):
        '''
        :param input_shape: Input shape; should be [batch_size, 1, width, F]
        '''
        features = input_shape[-1] # number of feature channels in input map
        self.weights = self.add_weight(
            shape=[features],
            initializer='random_normal', # TODO: possibly change this later?
            trainable=True,
            name=f'interp_{self.layer_number}'
        )

        # Scale weights to [0, 1] and create diagonal matrix
        scaled_weights = tf.nn.sigmoid(self.weights)
        weights_mat = tf.expand_dims(tf.linalg.diag(scaled_weights), axis=0)

        # Get counter weights (representing DS layer on other side of skip connection) and create diagonal matrix
        counter_weights = 1.0 - scaled_weights
        counter_weights_mat = tf.expand_dims(tf.linalg.diag(counter_weights), axis=0)

        # Concatenate the weights + counter weights along axis 0 to generate 2FxF weight matrix
        # and expand dims to include batch_size dimension in preparation for conv, to get shape (batch_size=1, channels=2, F, F)
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



