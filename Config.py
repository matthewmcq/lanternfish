import numpy as np


# stolen from Wave-U-Net model (for now)
def cfg():
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
                    'epochs': 10,
                    'max_songs': 86, # 86 = all songs
                    'max_samples_per_song': 420, # 
                    'num_init_filters': 25, # 

                    }
    
    return model_config
