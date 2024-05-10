def cfg():
    # Base configuration
    model_config = {'num_layers' : 12, # How many U-Net layers
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    'num_init_filters': 24, # THIS MUST BE DIVISIBLE BY 4 IF USING DUALWAVELETUNET


                    'learning_rate': 1e-4, # determine's the model's learning rate
                    'validation_split': 0.2, # determines what % of training data is used for validation
                    'channels': 1,
                    'num_coeffs': 16384, # Number of audio samples/detail coefficients per input; currently 220500 for 10 sec audio snippets (our equivalent of num_frames from Wave-U-Net)
                    'wavelet_depth': 2,
                    'batch_size' : 16, # Batch size
                    'epochs': 10,
                    'max_songs': 106, # 86 = all songs vox, 84 = all songs bass, 106 = all songs drumkit
                    'max_samples_per_song': 600, #

                    'l1_reg': 1e-11, # L1 regularization -> sparse
                    'l2_reg': 1e-12, # L2 regularization -> non-sparse

                    'lambda_vec': [1],
                    'lambda_11': 1,
                    'lambda_12': 1,
                    }

    return model_config

def cfg_retrain():
    # Base configuration
    model_config = {'num_layers' : 12, # How many U-Net layers
                    'filter_size' : 15, # For Wave-U-Net: Filter size of conv in downsampling block
                    'merge_filter_size' : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    'num_init_filters': 24, # THIS MUST BE DIVISIBLE BY 4 IF USING DUALWAVELETUNET


                    'learning_rate': 4e-5, # determine's the model's learning rate
                    'validation_split': 0.2, # determines what % of training data is used for validation
                    'channels': 1,
                    'num_coeffs': 16384, # Number of audio samples/detail coefficients per input; currently 220500 for 10 sec audio snippets (our equivalent of num_frames from Wave-U-Net)
                    'wavelet_depth': 2,
                    'batch_size' : 32, # Batch size
                    'epochs': 10,
                    'max_songs': 86, # 86 = all songs
                    'max_samples_per_song': 700, #

                    'l1_reg': 1e-12, # L1 regularization -> sparse
                    'l2_reg': 1e-11, # L2 regularization -> non-sparse

                    'lambda_vec': [1],
                    'lambda_11': 1,
                    'lambda_12': 1,
                    }

    return model_config
