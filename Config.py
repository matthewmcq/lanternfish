import numpy as np


# stolen from Wave-U-Net model (for now)
def cfg():
    # Base configuration
    model_config = {'num_layers' : 8, # How many U-Net layers
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    'input_filter_size' : 15, # For Wave-U-Net: Filter size of first convolution in first downsampling block
                    'output_filter_size': 1, # For Wave-U-Net: Filter size of convolution in the output layer
                    
                    'num_coeffs': 32768, # Number of audio samples/detail coefficients per input; currently 220500 for 10 sec audio snippets (our equivalent of num_frames from Wave-U-Net)
                    'wavelet_depth': 4,
                    'batch_size' : 40, # Batch size
                    'channels': 1, # 1 => mono, 2 => stereo -- currently only supporting mono to reduce complexity
                    'epochs': 10,
                    'max_songs': 1, # 86 = all songs
                    'max_samples_per_song': 100, # bring down to 200 so there is a more even distribution of samples per song
                    'num_init_filters': 15, # 

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
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
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
