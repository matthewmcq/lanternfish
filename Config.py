import numpy as np

def cfg():
    # Base configuration
    model_config = {'num_layers' : 12, # How many U-Net layers
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    'input_filter_size' : 15, # For Wave-U-Net: Filter size of first convolution in first downsampling block
                    'output_filter_size': 1, # For Wave-U-Net: Filter size of convolution in the output layer

                    'prelu': False, # determines if model is build with PReLU activations. False => leaky_relu
                    'learning_rate': 1e-4, # determine's the model's learning rate
                    'validation_split': 0.2, # determines what % of training data is used for validation

                    'num_coeffs': 32768, # Number of audio samples/detail coefficients per input; currently 220500 for 10 sec audio snippets (our equivalent of num_frames from Wave-U-Net)
                    'wavelet_depth': 3,
                    'batch_size' : 16, # Batch size
                    'channels': 1, # 1 => mono, 2 => stereo -- currently only supporting mono to reduce complexity
                    'epochs': 1000,
                    'max_songs': 86, # 86 = all songs
                    'max_samples_per_song': 3, #
                    'num_init_filters': 24, # THIS MUST BE DIVISIBLE BY 4 IF USING DUALWAVELETUNET

                    'l1_reg': 1e-12, # L1 regularization -> sparse
                    'l2_reg': 1e-11, # L2 regularization -> non-sparse

                    'lambda_vec': [1, 1,  1],
                    'lambda_11': 1,
                    'lambda_12': 1,
                    }

    return model_config

# stolen from Wave-U-Net model (for now) 
def cfg30():
    # Base configuration
    model_config = {'num_layers' : 13, # How many U-Net layers
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    'input_filter_size' : 15, # For Wave-U-Net: Filter size of first convolution in first downsampling block
                    'output_filter_size': 1, # For Wave-U-Net: Filter size of convolution in the output layer
                    
                    'num_coeffs': 32768, # Number of audio samples/detail coefficients per input; currently 220500 for 10 sec audio snippets (our equivalent of num_frames from Wave-U-Net)
                    'wavelet_depth': 4,
                    'batch_size' : 40, # Batch size
                    'channels': 1, # 1 => mono, 2 => stereo -- currently only supporting mono to reduce complexity
                    'epochs': 10,
                    'max_songs': 86, # 86 = all songs
                    'max_samples_per_song': 400, # bring down to 200 so there is a more even distribution of samples per song
                    'num_init_filters': 30, # 

                    'l1_reg': 1e-8, # L1 regularization
                    'l2_reg': 1e-7, # L2 regularization

                    'lambda_vec': [40, 2.5, 0.3, 0.2],
                    'lambda_11': 1,
                    'lambda_12': 0.25,
                    }
    
    return model_config

def _cfg(): # this one works, but takes forever to train, so I'm using the one above
    # Base configuration
    model_config = {'num_layers' : 13, # How many U-Net layers
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    'input_filter_size' : 15, # For Wave-U-Net: Filter size of first convolution in first downsampling block
                    'output_filter_size': 1, # For Wave-U-Net: Filter size of convolution in the output layer
                    
                    'num_coeffs': 32768, # Number of audio samples/detail coefficients per input; currently 220500 for 10 sec audio snippets (our equivalent of num_frames from Wave-U-Net)
                    'wavelet_depth': 4,
                    'batch_size' : 16, # Batch size
                    'channels': 1, # 1 => mono, 2 => stereo -- currently only supporting mono to reduce complexity
                    'epochs': 7,
                    'max_songs': 86, # 86 = all songs
                    'max_samples_per_song': 420, # 
                    'num_init_filters': 25, # 

                    'l1_reg': 1e-5, # L1 regularization
                    'l2_reg': 1e-4, # L2 regularization

                    'lambda_vec': [40, 2.5, 0.3, 0.2],
                    'lambda_11': 1,
                    'lambda_12': 0.25,
                    }
    
    return model_config

def test_saving():
    # Base configuration
    model_config = {'num_layers' : 5, # How many U-Net layers
                    'filter_size' : 11, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 7, # For Wave-U-Net: Filter size of conv in upsampling block
                    'input_filter_size' : 15, # For Wave-U-Net: Filter size of first convolution in first downsampling block
                    'output_filter_size': 1, # For Wave-U-Net: Filter size of convolution in the output layer
                    
                    'num_coeffs': 32768, # Number of audio samples/detail coefficients per input; currently 220500 for 10 sec audio snippets (our equivalent of num_frames from Wave-U-Net)
                    'wavelet_depth': 4,
                    'batch_size' : 16, # Batch size
                    'channels': 1, # 1 => mono, 2 => stereo -- currently only supporting mono to reduce complexity
                    'epochs': 1,
                    'max_songs': 86, # 86 = all songs
                    'max_samples_per_song': 2, # 
                    'num_init_filters': 15, # 

                    'l1_reg': 1e-8, # L1 regularization
                    'l2_reg': 1e-9, # L2 regularization

                    'lambda_vec': [40, 2.5, 0.3, 0.2],
                    'lambda_11': 1,
                    'lambda_12': 0.25,
                    }
    
    return model_config
