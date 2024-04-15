import tensorflow as tf
import numpy as np
import librosa
import sys

# from Main import BATCH_PARAMS
sys.path.append('..')
from Main import BATCH_PARAMS


class WaveletUNet(tf.keras.Model):
    def __init__(self, model_config):
        super().__init__()
        self.num_coeffs = model_config['num_coeffs']
        self.wavelet_depth = model_config['wavelet_depth']
        # self.batch_size = model_config['batch_size']
        self.channels = model_config['channels']
        self.num_layers = model_config['num_layers']
        self.num_init_filters = model_config['num_init_filters']
        self.filter_size = model_config['filter_size']

        self.input_shape = (self.wavelet_depth+1, self.num_coeffs, self.channels)

    def call(self, inputs):
        current_layer = inputs
        current_layer = self.get_output(current_layer)
        return current_layer

        

    def get_output(self, inputs):
        current_layer = tf.keras.layers.Input(shape=self.input_shape)(inputs)
        # current_layer = inputs

        current_layer = tf.keras.layers.Lambda(lambda x: x[0])(current_layer)
        
        enc_outputs = list()

        for i in range(self.num_layers):
            current_layer = tf.keras.layers.Conv1D(self.num_init_filters + (self.num_init_filters * i), self.filter_size, strides=1, activation='leaky_relu', padding='same')(current_layer)
            enc_outputs.append(current_layer)
            current_layer = current_layer[:,::2,:]

        for i in range(self.num_layers):
            current_layer = tf.keras.layers.Conv1D(self.num_init_filters + (self.num_init_filters * (self.num_layers - i - 1)), self.filter_size, strides=1, activation='leaky_relu', padding='same')(current_layer)
            current_layer = tf.keras.layers.UpSampling1D(2)(current_layer)
            # current_layer = self.crop(current_layer, current_layer.shape.as_list(), enc_outputs[-i-1].shape)
            target_shape = enc_outputs[-i-1].shape
            current_layer = tf.keras.layers.Lambda(
                lambda x: tf.image.crop_to_bounding_box(x, 
                                                        offset_height=0, 
                                                        offset_width=0, 
                                                        target_height=target_shape[1], 
                                                        target_width=target_shape[2]) ,
                                                        output_shape=(target_shape[1], target_shape[2]))(current_layer)
            current_layer = tf.keras.layers.Concatenate(axis=1)([current_layer, enc_outputs[-i-1]])

        current_layer = tf.keras.layers.Conv1D(self.num_coeffs * self.channels, self.filter_size, strides=1, activation='sigmoid', padding='same')(current_layer)
        return current_layer
    
    # def compile(self, optimizer, loss, metrics):
    #     self.optimizer = optimizer
    #     self.loss_function = loss
    #     self.accuracy_function = metrics[0]

    def crop(tensor, shape, target_shape, match_feature_dim=True):
        '''
        Crops a 3D tensor [batch_size, width, channels] along the width axes to a target shape.
        Performs a centre crop. If the dimension difference is uneven, crop last dimensions first.
        :param tensor: 4D tensor [batch_size, width, height, channels] that should be cropped. 
        :param target_shape: Target shape (4D tensor) that the tensor should be cropped to
        :return: Cropped tensor
        '''
        # shape = np.array(tensor.shape.as_list())
        diff = shape - np.array(target_shape)
        assert(diff[0] == 0 and (diff[2] == 0 or not match_feature_dim)) # Only width axis can differ
        if (diff[1] % 2 != 0):
            print("WARNING: Cropping with uneven number of extra entries on one side")
        assert diff[1] >= 0 # Only positive difference allowed
        if diff[1] == 0:
            return tensor
        crop_start = diff // 2
        crop_end = diff - crop_start

        return tensor[:,crop_start[1]:-crop_end[1],:]

def train(model, train_data, epochs, batch_size):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    
    model.fit(train_data, epochs=epochs, batch_size=batch_size)

    return model