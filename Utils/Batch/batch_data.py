import numpy as np
import tensorflow as tf
import librosa
import os
import soundfile as sf
import numpy as np
import pywt
import cv2
from Wavelets import WaveData, SR
import generate_examples as ge

def generate_pairs(path_to_song: str, stem_type: str, level: int =12) -> tuple:
    '''
    params:
    path_to_song: str, path to the song folder
    stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    level: int, level of wavelet decomposition
    '''
    
    # call Wavelets.makeWaveDict() to get the wavelet dictionary
    train_dict = WaveData.makeWaveDict(path_to_song + '/y_train')
    true_dict = WaveData.makeWaveDict(path_to_song + '/y_true')

    # call make_test_set() to get the test set
    y_train, y_true = make_test_set(train_dict, true_dict, stem_type, path_to_song, level)

    # convert to tensors
    y_train = tf.convert_to_tensor(y_train)
    y_true = tf.convert_to_tensor(y_true)

    return y_train, y_true


def make_test_set(train_dict: dict, true_dict: dict, stem_type: str, path_to_song: str, level: int=12) -> tuple:
    '''
    params:
    train_dict: dict, dictionary of wavelet data for training
    true_dict: dict, dictionary of true wavelet data
    stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    path_to_song: str, path to the song folder
    level: int, level of wavelet decomposition
    '''

    y_train = []
    y_true = []
    for key in train_dict.keys():
        train = train_dict[key]
        index = int(key.split("_")[-1][0])
        assert int(index) >= 0
        assert index != ""
        true_key = f"{stem_type}_{index}_pred.wav"
        true_key = path_to_song + "/y_true/" + true_key
        
        WaveData.getWaveletTransform(train_dict, key, level)
        WaveData.getWaveletTransform(true_dict, true_key, level)
        true = true_dict[true_key]
        train_tensor = train.tensor_coeffs
        true_tensor = true.tensor_coeffs
        y_train.append(train_tensor)
        y_true.append(true_tensor)

    return y_train, y_true

def batch_data(path_to_training: str, stem_type: str, level: int =12, batch_size: int =8, max_songs: int =2, max_samples_per_song: int =10) -> tuple:
    '''
    params:
    path_to_training: str, path to the training data
    stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    level: int, level of wavelet decomposition
    batch_size: int, size of the batch
    max_songs: int, maximum number of songs to use
    max_samples_per_song: int, maximum number of samples per song
    return: tuple, batch of training data
    '''
    if path_to_training[-1] != '/':
        path_to_training += '/'

    # find all songs in the training data
    songs = os.listdir(path_to_training + stem_type)

    # limit the number of songs -- choose randomly
    if len(songs) > max_songs:
        songs = np.random.choice(songs, max_songs, replace=False)


    # generate pairs for each song
    y_train = []
    y_true = []
    for song in songs:
        # generate pairs for each song
        path_to_song = path_to_training + stem_type + '/' + song + '/'
        train, true = generate_pairs(path_to_song, stem_type, level)

        # limit the number of samples per song
        if len(train) > max_samples_per_song:
            indices_to_keep = np.random.choice(len(train), max_samples_per_song, replace=False)

            train = [train[i] for i in indices_to_keep]
            true = [true[i] for i in indices_to_keep]

        # add to the list
        y_train.append(train)
        y_true.append(true)

    # convert to tensors
    y_train = tf.convert_to_tensor(y_train)
    y_true = tf.convert_to_tensor(y_true)

    # randomize the data making sure to keep the pairs together, and split into batches
    indices = np.random.permutation(len(y_train))
    y_train = y_train[indices]
    y_true = y_true[indices]


    # make batches
    batch = []
    for i in range(0, len(y_train), batch_size):
        batch.append((y_train[i:i+batch_size], y_true[i:i+batch_size]))

    return batch
