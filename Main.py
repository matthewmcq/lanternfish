import librosa
import Utils.Batch.generate_examples
import Utils.Batch.batch_data
import Utils.Plot
import tensorflow as tf
from Models.WaveletUNet import WaveletUNet
import Config as cfg
import Train
import numpy as np
import cv2
import pywt
import soundfile as sf
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
### DO NOT CHANGE ###
MEDLEY2_PATH = 'Datasets/MedleyDB/V2/'
MEDLEY1_PATH = 'Datasets/MedleyDB/V1/'
TRAIN_PATH = 'Datasets/TrainingData/'

## Set current stem type to process. Options are: 'vocals', 'drums', 'bass', 'midrange'
CURR_STEM_TYPE = 'vocals'
SR = 44100

def main_train():
    model_config = cfg()

    WAVELET_DEPTH = model_config['wavelet_depth'] # level of wavelet decomposition
    BATCH_SIZE = model_config['batch_size'] # number of samples per batch
    MAX_SONGS = model_config['max_songs'] # maximum number of songs to include in the batch
    MAX_SAMPLES_PER_SONG = model_config['max_samples_per_song'] # maximum number of samples per song to include in the batch

    ## Set the batch parameters, pass to batch_training_data()
    BATCH_PARAMS = (WAVELET_DEPTH, BATCH_SIZE, MAX_SONGS, MAX_SAMPLES_PER_SONG)


    ## set the batch size and epochs
    batch_size = model_config['batch_size']
    epochs = model_config['epochs']


    model = WaveletUNet(
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
    model(tf.random.normal(shape=(batch_size, model_config['num_coeffs'], WAVELET_DEPTH+1)))

    # print the model summary
    model.summary()

    dataset, validation_data, shape = Utils.Batch.batch_data.batch_training_data(*BATCH_PARAMS)

    loss =  Train.WaveletLoss(wavelet_level=model_config['wavelet_depth'], lambda_vec=model_config['lambda_vec'], lambda_11=model_config['lambda_11'], lambda_12=model_config['lambda_12'], name='wavelet_loss')

    ## train the model
    model = Train.train(model, model_config, loss, dataset, validation_data)

    model_name = f'matt_a100_newloss_depth2_{CURR_STEM_TYPE}_v6.keras'
    model.save(model_name)

def main_load(custom_objects=None):
    model_config = cfg()
    
    WAVELET_DEPTH = model_config['wavelet_depth'] # level of wavelet decomposition
    BATCH_SIZE = model_config['batch_size'] # number of samples per batch
    MAX_SONGS = model_config['max_songs'] # maximum number of songs to include in the batch
    MAX_SAMPLES_PER_SONG = model_config['max_samples_per_song'] # maximum number of samples per song to include in the batch
    
    # Set the batch parameters, pass to batch_training_data()
    BATCH_PARAMS = (WAVELET_DEPTH, BATCH_SIZE, MAX_SONGS, MAX_SAMPLES_PER_SONG)
    
    # batch the data for medleyDB
    # preprocess_medleydb(CURR_STEM_TYPE, clean=True)
    y_train, y_true, shape = Utils.Batch.batch_data.batch_training_data(*BATCH_PARAMS)
    
    model_name = f'wavelet_unet_model_nif{model_config["num_init_filters"]}_filter{model_config["filter_size"]}_layers{model_config["num_layers"]}.keras'
    
    loaded_model = tf.keras.models.load_model(model_name, custom_objects=custom_objects)
    
    # define a dummy input to build the model
    loaded_model(tf.random.normal(shape=(model_config['batch_size'], model_config['num_coeffs'], WAVELET_DEPTH+1)))
    get_prediction(loaded_model, y_train, y_true)

def get_prediction(loaded_model, y_train, y_true):
    for i in range(20):
        prediction = loaded_model.predict(tf.expand_dims(y_train[i], axis=0))[0]

        true = np.transpose(y_true[i], (1, 0))
        a3_true, d3_true, d2_true = true
        sum_all_true = a3_true + d3_true + d2_true

        train = np.transpose(y_train[i], (1, 0))
        a3_train, d3_train, d2_train = train
        sum_all_train = a3_train + d3_train + d2_train

        prediction = np.squeeze(prediction, axis=-1)

        if not os.path.exists('audiofiles'):
            os.mkdir('audiofiles')
        if not os.path.exists(f'audiofiles/sample{i}'):
            os.mkdir(f'audiofiles/sample{i}')

        # Low-pass filter the audio signal
        predict_filtered = low_pass_filter(prediction, 11000)

        Utils.Plot.display_waveforms(true, prediction, i)

        Utils.Plot.display_stft(sum_all_train, i, 'train')
        Utils.Plot.display_stft(sum_all_true, i, 'true')
        Utils.Plot.display_stft(prediction, i, 'predict')
        Utils.Plot.display_stft(predict_filtered, i, 'predict_filtered')

        sf.write(f'audiofiles/sample{i}/train.wav', sum_all_train, SR)
        sf.write(f'audiofiles/sample{i}/true.wav', sum_all_true, SR)
        sf.write(f'audiofiles/sample{i}/predict.wav', prediction, SR)
        sf.write(f'audiofiles/sample{i}/predict_filtered.wav', predict_filtered, SR)


def low_pass_filter(audio, cutoff):
    b, a = signal.butter(3, cutoff / (SR / 2), btype='low', analog=False)
    return signal.filtfilt(b, a, audio, axis=-1)
