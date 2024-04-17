import tensorflow as tf
import numpy as np
import librosa
import sys

######################################################################################################################################################################

########## GENERAL NOTES FOR HYPERPARAMETER TUNING: ########## -Matt 4/16 (FEEL FREE TO ADD TO THIS LIST AND/OR MOVE IT TO A DIFFERENT FILE)

### EXTREMELY IMPORTANT: ALL LOSS VALUES ARE NOT EXACTLY APPLES-TO-APPLES BECAUSE THE DATA IS SELECTED RANDOMLY EACH TIME

## ALSO, I'm running benchmarking tests on sets of ~3.5k examples, the full dataset is over 100k examples (zamn...), so the results will be different when run on the full dataset

## CROPPING ISSUE HAS BEEN FIXED!!

## num_layers is the number of U-Net layers, but incresing will very quickly increase the number of parameters -- requires the crop layer to be adjusted

## Also, increasing num_init_filters will increase the number of parameters, but will also increase the complexity of the model -- does not change crop

## filter_size is the size of the convolutional filters in the downsampling and upsampling blocks -- increasing will increase the number of parameters -- does not change crop

## other ways to increase complexity: 
#   -increase wavelet_depth, 
#   -increase batch size, 
#   -increase number of epochs, 
#   -switch to a different loss function, 
#   -switch from leaky_relu to prelu
#   -add batch normalization layers (maybe... could also mess with normalizing the whole data or using LayerNormalization)

######################################################################################################################################################################

########## BENCHMARKING: ##########

## (THIS IS ALL PRE CROPPING FIX), just wanted to have a baseline to compare to after the crop fix and for the sake of note-taking

### BATCHNORM: -- Test with and without BatchNorm (BN) and compare (currently running) -- might play around with batch size: ###

# ~ 18 min per epoch -- Notes: decent results, but still not great. Loss is decreasing, but starts to plateau after 3 epochs.
# -- Might just need to run for longer...could be local min, or could be that the model is not complex enough

# Stats: WITH BN: batches 84, bs 32, init_filters 12, layers 12, filter_size 16, wavelet_depth 4 -- loss (MSE): ep1 0.0159, ep2 0.0153, ep3 0.0153
# Stats: W/O  BN: batches 84, bs 32, init_filters 12, layers 12, filter_size 16, wavelet_depth 4 -- loss (MSE): ep1 0.0133, ep2 0.0129, ep3 0.0128

### NUM_INIT_FILTERS: -- Increase num_init_filters to 24 and compare: ###

# ~ 45 min per epoch -- Notes: Takes A LOT longer to run, and the initial results are not much better than 12 init_filters. 
# -- curious to see how it performs with more layers and/or wavelet depth + letting it run overnight
# -- Also, might be worth trying to increase the filter size or playing around with batch size

# Stats: W/O  BN: batches 80, bs 32, init_filters 24, layers 12, filter_size 16, wavelet_depth 4 -- loss (MSE): ep1 0.0133, ep2 -.----, ep3 -.----

### NUM_LAYERS: -- Decreased num_layers to 8 and compare (requires wayyy less crop) ### 

# ~ 5 min per epoch -- Notes: Takes way less time to run, and the results are suspiciously good.
# -- this is testing to see if the crop is the issue or if the model is just not complex enough
# -- WE NEED A WAY TO VISUALIZE THE OUTPUTS TO SEE WHAT IS HAPPENING
# -- I think if I let this run for a while, it will start to plateau really hard, so more layers is probably better but seems worse for earlier epochs (could be wrong though)

# Stats: W/O  BN: batches 80, bs 32, init_filters 12, layers 8, filter_size 16, wavelet_depth 4 -- loss (MSE): ep1 0.0097, ep2 0.0093, ep3 0.0092

######################################################################################################################################################################

####### POST CROPPING FIX: #######

## DEFAULT (w extra data): 12 layers, 12 init_filters, 16 filter_size, 4 wavelet_depth, 32 batch_size

# ~ 13 min per epoch -- Notes: Seems promising, not as groundbreaking as I had hoped after fixing the cropping issue, but gives a lot of hope for playing with more/fewer layers

# Stats: W/O  BN: batches 99, bs 32, init_filters 12, layers 12, filter_size 16, wavelet_depth 4 -- loss (MSE): ep1 0.0146, ep2 0.0131, ep3 0.0130, ep4 0.0130
#                                                                                                    @batch 84: ep1 0.0152

### TODO BATCH_SIZE: -- Decrease batch size to 4, 8, 16 and compare, then potentially increase to 64 and compare

## Batch size 4: (DO NOT USE) ~ 13 min per epoch -- Notes: started off strong but became unstable and gradient exploded halfway through first epoch

## Batch size 16: ~ 12 min per epoch -- Notes: Extremely Promising, loss is decreasing at a good rate, gradient is stable, think this might be the sweet spot

# Stats: W/O  BN: batches 179, bs 16, init_filters 12, layers 12, filter_size 16, wavelet_depth 4 -- loss (MSE): ep1 0.0124, ep2 0.0120, ep3 0.0119

### TODO: NUM_LAYERS: -- Decrease to 6 or 8, Increase num_layers to 16, 20, and 24 and compare

# Num_layers 8:  ~ 5.5 min per epoch -- Notes: Similar to before fixing the crop, weirdly good results, but loss increases steadily 
#                                              up to max that is still lower than best results for earlier versions instead of decreasing
#                                              frankly, I am just confused by how much better this is than 12 layers... 
#                                              maybe it just works for our data and we're biased to trust the Wave-UNet guys?
#                                              My theory is that maybe the training examples with no audio are messing with the model

# Stats: W/O  BN: batches 179, bs 16, init_filters 12, layers 8, filter_size 16, wavelet_depth 4 -- loss (MSE): ep1 0.0103, ep2  0.0099, ep3  -.---


# Num_layers 10: ~ 9 min per epoch Notes: converged to loss of 0.0131 after 6 epochs, need more complexity

# Stats: W/O  BN: batches 180, bs 16, init_filters 12, layers 10, filter_size 16, wavelet_depth 4 -- loss (MSE): ep1 0.0136, ep2 0.0127, ep3 0.0126, ep4 0.0179, ep5 0.0131, ep6 0.0131 (same for rest)

##-- 
# Num_layers 20: ~ 20 min per epoch (letting run overnight)

### TODO: FILTER_SIZE: -- Increase filter_size to 20, 24, and 32 and compare

### TODO: DATA SIZE: -- Double/triple the data and compare

### TODO: -- Increase wavelet_depth to 7 or 8 and compare

### TODO: IMPORTANT -- WRITE A NEW NOT SHITASS LOSS FUNCTION THAT ISNT FUCKING MSE

######################################################################################################################################################################

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

        # # Create batch normalization layers for downsampling blocks
        # self.DS_batch_norm = {}
        # for i in range(self.num_layers):
        #     block_name = f'ds{i+1}'
        #     self.DS_batch_norm[block_name] = tf.keras.layers.BatchNormalization(axis=1, name=f'{block_name}_batch_norm')

        # Create bottle neck
        self.bottle_neck = tf.keras.layers.Conv1D(self.num_init_filters + (self.num_init_filters * self.num_layers), self.filter_size, activation='leaky_relu', padding='same', strides=1, name='bottle_neck')
        
        # Create upsampling blocks
        self.upsampling_blocks = {}
        for i in range(self.num_layers):
            block_name = f'us{i+1}'
            num_filters = self.num_init_filters + (self.num_init_filters * (self.num_layers - i - 1))
            self.upsampling_blocks[block_name] = UpsamplingLayer(num_filters, self.filter_size, name=block_name)
        
        # # Create batch normalization layers for upsampling blocks
        # self.US_batch_norm = {}
        # for i in range(self.num_layers):
        #     block_name = f'us{i+1}'
        #     self.US_batch_norm[block_name] = tf.keras.layers.BatchNormalization(axis=1, name=f'{block_name}_batch_norm')


        # Cropping layer
        # self.last_crop = tf.keras.layers.Cropping1D(cropping=(956, 956), name='last_crop')  # -- FOR 12 LAYERS
        # self.last_crop = tf.keras.layers.Cropping1D(cropping=(444, 444), name='last_crop') # -- FOR 10 LAYERS
        # self.last_crop = tf.keras.layers.Cropping1D(cropping=(60, 60), name='last_crop')  # -- FOR 8 LAYERS


        # Final convolution layer
        self.output_conv = tf.keras.layers.Conv1D(self.wavelet_depth, 1, activation='tanh', name='output_conv')
        
        super().build(input_shape)


    def call(self, inputs):
        
        current_layer = inputs 

        enc_outputs = list()

        # Downsampling path
        for i in range(self.num_layers):

            # Apply downsampling block
            block_name = f'ds{i+1}'
            block = self.downsampling_blocks[block_name]

            current_layer = block(current_layer)
            
            # # Apply batch normalization
            # if is_training:
            #     batch_norm = self.DS_batch_norm[block_name]
            #     current_layer = batch_norm(current_layer)

            # Save for skip connections
            enc_outputs.append(current_layer)
            
            # Decimation step
            current_layer = current_layer[:, ::2, :]
            

        # Bottle neck
        current_layer = self.bottle_neck(current_layer)

        # Upsampling path
        for i in range(self.num_layers):

            # Apply upsampling block
            block_name = f'us{self.num_layers - i}'
            block = self.upsampling_blocks[block_name]

            current_layer = block(current_layer)

            # Get skip connection
            skip_conn = enc_outputs[-i-1]
            
            # Pad if necessary
            desired_shape = skip_conn.shape

            ### OLD PADDING METHOD -- pad skip with zeros
            # if current_layer.shape[1] != desired_shape[1]:
            #     pad = current_layer.shape[1] - desired_shape[1]
            #     skip_conn = tf.pad(skip_conn, [[0, 0], [0, pad], [0, 0]], 'CONSTANT')

            ### NEW CROPPING METHOD -- crop current_layer to match skip_conn
            if current_layer.shape[1] != desired_shape[1]:
                diff = current_layer.shape[1] - desired_shape[1]
                crop_start = diff // 2
                crop_end = crop_start + desired_shape[1]
                current_layer = tf.slice(current_layer, [0, crop_start, 0], [-1, desired_shape[1], -1])


            # Concatenate with skip connection
            current_layer = tf.keras.layers.Concatenate()([current_layer, skip_conn])

            # # Apply batch normalization
            # if is_training:
            #     batch_norm = self.US_batch_norm[block_name]
            #     current_layer = batch_norm(current_layer)
            
        
        # Final convolution layer, tanh activation
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