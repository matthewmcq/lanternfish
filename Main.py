import Utils.Batch.generate_examples
import Utils.Batch.batch_data
import Utils.Plot
import tensorflow as tf
import Models.wavelet_unet
import Models.wavelet_unet_prelu
import Config as cfg
from Train import train, WaveletLoss, WaveletLoss2
import numpy as np
import cv2
import pywt
import soundfile as sf
import matplotlib.pyplot as plt
### DO NOT CHANGE ###
MEDLEY2_PATH = 'Datasets/MedleyDB/V2/'
MEDLEY1_PATH = 'Datasets/MedleyDB/V1/'
TRAIN_PATH = 'Datasets/TrainingData/'

## Set current stem type to process. Options are: 'vocals', 'drums', 'bass', 'midrange'
CURR_STEM_TYPE = 'vocals'




def inverseWaveletReshape(tensor_coeffs, shape, wavelet_depth):
    """
    Reverse the wavelet transform and downscale the tensor coefficients to match the original shape.

    Args:
        tensor_coeffs (tf.Tensor): The tensor of wavelet coefficients, with shape (max_features, wavelet_depth + 1).
        shape (tuple): The original shape of the waveform.
        wavelet_depth (int): The depth of the wavelet decomposition.

    Returns:
        list: A list of tuples representing the downscaled wavelet coefficients.
    """
    # Convert the tensor to a NumPy array
    # coeffs = tensor_coeffs.numpy()
    coeffs = tensor_coeffs

    # Create a list to store the downscaled coefficients
    downscaled_coeffs = []

    # Iterate over the wavelet levels
    for level in range(wavelet_depth + 1):
        # Get the coefficients for the current level
        level_coeffs = coeffs[:, level]
        # print(f"level_coeffs: {level_coeffs.shape}")
        # print(f"level_coeffs: {level_coeffs.shape}")
        # interval = shape[level][0] // level_coeffs.shape[0]
        # replace = level_coeffs[::interval, :]

        # print(f"replace: {replace}")
        # print(f"replace.shape: {replace.shape}")
        

        # Reshape the coefficients to match the original shape
        # reshaped_coeffs = level_coeffs.reshape(shape[level])
        dsize = (shape[level][0], 1)
        # print(f"dsize: {dsize}")
        reshaped_coeffs = cv2.resize(level_coeffs.reshape(1, -1), dsize=dsize, interpolation=cv2.INTER_AREA).flatten()
        # print(f"reshaped_coeffs.shape: {reshaped_coeffs.shape}")
        # print(f"reshaped_coeffs: {reshaped_coeffs}")

        # Collapse the noisy lower LOD detail and approximation coefficients
        # collapsed_coeffs = np.mean(reshaped_coeffs, axis=1)
        # collapsed_coeffs = np.median(reshaped_coeffs, axis=1)

        # Append the collapsed coefficients to the list
        downscaled_coeffs.append(reshaped_coeffs)

    # print(f"downscaled_coeffs: {downscaled_coeffs}")
    # downscaled_coeffs = np.array(downscaled_coeffs).flatten()
    return downscaled_coeffs

def preprocess_medleydb(stem_type: str, clean: bool =False, sample_length=65536) -> None:
    '''
    Preprocess the MedleyDB dataset to generate training data

    params:
    - stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    - clean: bool, flag to clean the training data

    return: None
    '''

    ## call clean_training_data() first to clean the training data if something goes wrong
    if clean: 
        Utils.Batch.generate_examples.clean_training_data(TRAIN_PATH, stem_type)

    ## call generate_examples() to generate the examples
    Utils.Batch.generate_examples.generate_data(MEDLEY1_PATH, TRAIN_PATH, stem_type, sample_length) ## -- WORKS!
    Utils.Batch.generate_examples.generate_data(MEDLEY2_PATH, TRAIN_PATH, stem_type, sample_length) ## -- WORKS!


def batch_training_data(level: int = 12, batch_size: int = 8, max_songs: int = 2, max_samples_per_song: int = 10, num_features: int=65536) -> tf.data.Dataset:
    '''
    Batch the wavelet data for training

    params:
    - level: int, level of wavelet decomposition
    - batch_size: int, number of samples per batch
    - max_songs: int, maximum number of songs to include in the batch
    - max_samples_per_song: int, maximum number of samples per song

    return: 
    - tf.data.Dataset, batched wavelet data
    '''
    ## call batch_wavelets() to batch the wavelet data
    y_train, y_true, shape = Utils.Batch.batch_data.batch_wavelets_dataset(TRAIN_PATH, 'vocals', level, batch_size, max_songs, max_samples_per_song, num_features)
    
    return y_train, y_true, shape

def calculate_average_features(y_train):
    num_examples, num_timesteps, num_levels = y_train.shape
    average_features = np.zeros(num_levels)
    
    for level in range(num_levels):
        average_features[level] = np.mean(y_train[:, :, level])
    
    return average_features

def plot_feature_distribution(y_train):
    num_examples, num_timesteps, num_levels = y_train.shape
    
    for level in range(num_levels):
        plt.figure()
        plt.hist(y_train[:, :, level].numpy().flatten(), bins=50)
        plt.xlabel(f'Feature Value (Level {level})')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Features (Level {level})')
        plt.show()

def aggregate_dataset(wavelet_depth=3, batch_size=1, max_songs=86, max_samples_per_song=400):
    y_train, y_true, shape = batch_training_data(wavelet_depth, batch_size, max_songs, max_samples_per_song)
    
    average_features_train = np.zeros(wavelet_depth + 1)
    
    for level in range(wavelet_depth + 1):
        average_features_train[level] = np.mean(np.abs(y_train[:, :, level]))
    
    print(average_features_train)
        
    average_features_true = np.zeros(wavelet_depth + 1)
    
    for level in range(wavelet_depth + 1):
        average_features_true[level] = np.mean(np.abs(y_true[:, :, level]))
    
    
    print(average_features_true)
    # plot_feature_distribution(y_train)
    # plot_feature_distribution(y_true)
    
    
    

def main():

    # model_config = cfg.test_saving()
    # aggregate_dataset()
    
    model_config = cfg.cfg()

    ## Set the parameters -- might want to move to Config.py later
    WAVELET_DEPTH = model_config['wavelet_depth'] # level of wavelet decomposition
    BATCH_SIZE = model_config['batch_size'] # number of samples per batch
    MAX_SONGS = model_config['max_songs'] # maximum number of songs to include in the batch
    MAX_SAMPLES_PER_SONG = model_config['max_samples_per_song'] # maximum number of samples per song to include in the batch

    ## Set the batch parameters, pass to batch_training_data()
    BATCH_PARAMS = (WAVELET_DEPTH, BATCH_SIZE, MAX_SONGS, MAX_SAMPLES_PER_SONG)

    ## batch the data for medleyDB
    # preprocess_medleydb(CURR_STEM_TYPE, clean=True)

    ## set the batch size and epochs
    batch_size = model_config['batch_size']
    epochs = model_config['epochs']

    ## test that generate_pairs() works
    train_dataset, val_dataset, shape = batch_training_data(*BATCH_PARAMS)

    print("y_train shape:", shape)
    
    # print(y_train)
    # print(y_true)
    
    
    # loaded_model = tf.keras.models.load_model('trained_models/2d/wavelet_unet_model_nif40_filters21_nl13_2D.keras')
    # loaded_model(tf.random.normal(shape=(batch_size, model_config['num_coeffs'], WAVELET_DEPTH+1, 1)))

    # for i in range(10):
    #     predict_train_0 = loaded_model.predict(tf.expand_dims(y_train[i], axis=0))[0]
    #     # predict_true_0 = tf.expand_dims(y_true[i], axis=0).numpy()
    #     predict_true_0 = y_true[i].numpy()
    #     print(f"y_true_i.shape: {y_true[i].shape}")
    #     print(f"y_train_i.shape: {predict_train_0.shape}")
    #     # print(f"predict_train_0.shape: {predict_train_0.shape}")
    #     predict_train_0 = inverseWaveletReshape(predict_train_0, shape, model_config['wavelet_depth'])
    #     predict_true_0 = inverseWaveletReshape(predict_true_0, shape, model_config['wavelet_depth'])
    #     print(f"predict_true_0: {predict_true_0}")
    #     print(f"y_true_i: { tf.expand_dims(y_true[i], axis=0).numpy()}")
    #     print([c.shape for c in predict_true_0])
    #     # predict_true_0 = list(predict_true_0)
    #     # print(f"predict_train_0.shape no IWTReshape: {(predict_train_0)}")
    #     # output = pywt.waverec(predict_true_0, 'haar')
    #     output_true = pywt.waverec(predict_true_0, 'haar')
    #     output_train = pywt.waverec(predict_train_0, 'haar')
    #     # print(f"output in loop: {output_true}")
    #     # print(f"output.shape in loop: {output.shape}")
        
    #     sf.write(f'audiofiles/true{i}_2D.wav', output_true, 41000)
    #     sf.write(f'audiofiles/train{i}_2D.wav', output_train, 41000)
    #     # sf.write(f'true{i}.wav', y_true[i], SR//2)
    

    ## check the loss function for all zeros
    zero_train = tf.zeros_like(y_train)


    # wavelet_loss = WaveletLoss(
    #     shape=shape,
    #     wavelet_level=model_config['wavelet_depth'],
    #     lambda_vec=model_config['lambda_vec'],
    #     lambda_11=model_config['lambda_11'],
    #     lambda_12=model_config['lambda_12'],
    # )
    # loss = tf.keras.losses.MeanSquaredError()
    ## check default loss:
    loss = WaveletLoss( wavelet_level=model_config['wavelet_depth'], lambda_vec=model_config['lambda_vec'], lambda_11=model_config['lambda_11'], lambda_12=model_config['lambda_12'], name='wavelet_loss')
    # print("Default Loss with regularization:", loss(y_true, y_train))
    # print("Default Loss (All zeros):", loss(y_true, zero_train))
    # loss2 = WaveletLoss2( wavelet_level=model_config['wavelet_depth'], lambda_vec=model_config['lambda_vec'], lambda_11=model_config['lambda_11'], lambda_12=model_config['lambda_12'], name='wavelet_loss')
    # print("Default Loss with regularization:", loss2(y_true, y_train))
    # print("Default Loss (All zeros):", loss2(y_true, zero_train))

    if model_config['prelu']:
        model = Models.wavelet_unet_prelu.WaveletUNetPReLU(
            num_coeffs=model_config['num_coeffs'],
            wavelet_depth=model_config['wavelet_depth'],
            batch_size=model_config['batch_size'],
            channels=model_config['channels'],
            num_layers=model_config['num_layers'],
            num_init_filters=model_config['num_init_filters'],
            filter_size=model_config['filter_size'],
            merge_filter_size=model_config['merge_filter_size'],
            l1_reg=model_config['l1_reg'],
            l2_reg=model_config['l2_reg']
        )
    else:
        ## define the model
        model = Models.wavelet_unet.WaveletUNet(
            num_coeffs=model_config['num_coeffs'],
            wavelet_depth=model_config['wavelet_depth'],
            batch_size=model_config['batch_size'],
            channels=model_config['channels'],
            num_layers=model_config['num_layers'],
            num_init_filters=model_config['num_init_filters'],
            filter_size=model_config['filter_size'],
            merge_filter_size=model_config['merge_filter_size'],
            l1_reg=model_config['l1_reg'],
            l2_reg=model_config['l2_reg']
        )

    # define a dummy input to build the model
    model(tf.random.normal(shape=(batch_size, model_config['num_coeffs'], WAVELET_DEPTH+1, 1)))

    # print the model summary
    model.summary()
    ## train the model
    model = train(model, model_config, loss, train_dataset, val_dataset)

    model_name = f'wavelet_unet_model_nif{model_config["num_init_filters"]}_filters{model_config["filter_size"]}_nl{model_config["num_layers"]}_2D.keras'
    model.save(model_name)
    # model.save('wavelet_unet_model.h5')

    loaded_model = tf.keras.models.load_model(model_name)
    # loaded_model = tf.keras.models.load_model('wavelet_unet_model.h5')

@tf.keras.utils.register_keras_serializable()
class WaveletLoss(tf.keras.losses.Loss):
    def __init__(self, wavelet_level=4, lambda_vec=[40, 2.5, 0.3, 0.2], lambda_11=1, lambda_12=0.25, name='wavelet_loss',   l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.wavelet_level = wavelet_level
        self.lambda_vec = lambda_vec
        self.lambda_11 = lambda_11
        self.lambda_12 = lambda_12
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    # @tf.function
    def call(self, y_true, y_pred):
        # First index are approximation (midband) coefficients, second index are detail coefficients
        loss = self.lambda_11 * tf.math.reduce_mean(tf.math.square((y_true[:, :, 0, :] - y_pred[:, :, 0, :])))
        loss += self.lambda_12 * tf.math.reduce_mean(tf.math.square((y_true[:, :, 1, :] - y_pred[:, :, 1, :])))
        loss *= self.lambda_vec[0]

        # For levels 2 through second to last, take MAE of detail coefficients times lambda
        for i in range(2, self.wavelet_level):
            loss += self.lambda_vec[i-1] * tf.math.reduce_mean(tf.math.square((y_true[:, :, i, :] - y_pred[:, :, i, :])))

        # For the last level, take MSE of detail coefficients times lambda
        loss += self.lambda_vec[-1] * tf.math.reduce_mean(tf.math.square((y_true[:, :, -1, :] - y_pred[:, :, -1, :])))



        return loss

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
            name='bottleneck_conv',
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
            name='output_conv1',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        )

        self.output_conv2 = tf.keras.layers.Conv2D(
            1,
            (1, 1),
            activation='tanh',
            padding='same',
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
                if current_layer.shape[1] != desired_shape[1]:
                    diff = desired_shape[1] - current_layer.shape[1]
                    if diff >0:
                        pad_start = diff // 2
                        pad_end = diff - pad_start
                        current_layer = tf.pad(current_layer, [[0, 0], [pad_start, pad_end], [0, 0], [0,0]], mode='SYMMETRIC')
                    else:
                        diff = -diff
                        crop_start = diff // 2
                        current_layer = tf.slice(current_layer, [0, crop_start, 0, 0], [-1, desired_shape[1],-1, -1])


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
            if diff >0:
                pad_start = diff // 2
                pad_end = diff - pad_start
                current_layer = tf.pad(current_layer, [[0, 0], [pad_start, pad_end], [0, 0], [0,0]], mode='SYMMETRIC')
            else:
                diff = -diff
                crop_start = diff // 2
                current_layer = tf.slice(current_layer, [0, crop_start, 0, 0], [-1, desired_shape[1],-1, -1])


        current_layer = tf.keras.layers.Concatenate()([inputs, current_layer])
        # print(f"shape after final concat: {current_layer.shape}")

        # Final convolution layer, tanh activation
        current_layer = self.output_conv2(current_layer)

        # print(f"shape after final conv: {current_layer.shape}")
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
        self.conv = tf.keras.layers.Conv2D(
            self.num_filters,
            (self.filter_size, 1),
            activation='leaky_relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='upsampling_conv'
        )
        self.convtranspose = tf.keras.layers.Conv2DTranspose(
            self.num_filters,
            (self.filter_size, 1),
            strides=(self.strides, 1),
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
        self.conv = tf.keras.layers.Conv2D(
            self.num_filters,
            (self.filter_size, 1),
            activation='leaky_relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            activity_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='upsampling_conv_reshape'
        )
        super().build(input_shape)

    def call(self, inputs):
        weights_scaled = tf.nn.sigmoid(self.interp_weights)  # Constrain weights to [0, 1]
        counter_weights = 1.0 - weights_scaled
        # print("weights scaled shape", weights_scaled.shape)
        
        # print("inputs shape", inputs.shape)
        weights_scaled = tf.concat([tf.expand_dims(tf.linalg.diag(weights_scaled), axis=0), tf.expand_dims(tf.linalg.diag(counter_weights), axis=0)], axis=0)
        # print("weights scaled shape", weights_scaled.shape)
        weights_scaled = tf.expand_dims(weights_scaled, axis=0) # (1,2,C,C)
        weights_scaled = tf.transpose(weights_scaled, [1,0,2,3]) # (2,1,C,C)
        # print("weights scaled shape", weights_scaled.shape)
        # print(weights_scaled.shape)
        weights_scaled = tf.nn.conv2d(inputs, weights_scaled, strides=[1, 1, 1, 1], padding='SAME') #
        # print("weights scaled after conv", weights_scaled.shape)
        weights_scaled = tf.transpose(weights_scaled, [1,0,2,3])
        # print("weights scaled after transpose 1 ", weights_scaled.shape)
        
        inputs = tf.transpose(inputs, [1, 0, 2, 3])
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
        weights_scaled = tf.transpose(weights_scaled, [1, 0, 2, 3])
        # print("weights scaled after transpose 3", weights_scaled.shape)
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

if __name__ == '__main__':
    main()
