import numpy as np
import tensorflow as tf
import os
import numpy as np
import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import Wavelets
# import generate_examples as ge

def generate_pairs(path_to_song: str, stem_type: str, level: int =12, max_songs_per_stem=10) -> tuple:
    '''
    Generate pairs of wavelet data for training and true data

    params:
    - path_to_song: str, path to the song folder
    - stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    - level: int, level of wavelet decomposition

    return: 
    - tuple, pair of wavelet data for training and true data for the audio file
    '''

    ## TODO make random list of indices to select from that are of length max_songs_per_stem

    len_train = len(os.listdir(path_to_song + 'y_train/'))

    indices = np.random.choice(len_train, max_songs_per_stem, replace=True) # got issues with replace=False, so set to True for now
    
    # call Wavelets.makeWaveDict() to get the wavelet dictionary
    train_dict = Wavelets.makeWaveDict(path_to_song + 'y_train/', indices=indices)
    true_dict = Wavelets.makeWaveDict(path_to_song + 'y_true/', indices=None)

    # call make_test_set() to get the test set
    y_train, y_true, shape = make_test_set(train_dict, true_dict, stem_type, path_to_song, level, max_songs_per_stem)

    # convert to tensors
    y_train = tf.convert_to_tensor(y_train)
    y_true = tf.convert_to_tensor(y_true)

    return y_train, y_true, shape


def make_test_set(train_dict: dict, true_dict: dict, stem_type: str, path_to_song: str, level: int=12, max_songs_per_stem: int=10) -> tuple:
    '''
    Generate the test set for the wavelet data

    params:
    - train_dict: dict, dictionary of wavelet data for training
    - true_dict: dict, dictionary of true wavelet data
    - stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    - path_to_song: str, path to the song folder
    - level: int, level of wavelet decomposition

    return: 
    - tuple, pair of wavelet data arrays for training and true data for the full song folder
    '''

    y_train = []
    y_true = []

    song_name = path_to_song.split('/')[-2]

    train_shape = None

    ## select random max_songs_per_stem keys
    keys = list(train_dict.keys())
    if len(keys) > max_songs_per_stem:
        keys = np.random.choice(keys, max_songs_per_stem, replace=False)


    for key in tqdm.tqdm(keys, desc=f"Generating Test Set, Computing DWT for {song_name}", total=len(train_dict), leave=False):
        
        # get the true key
        train = train_dict[key]
        index = int(key.split("_")[-1][0])
        assert int(index) >= 0
        assert index != ""
        true_key = f"{stem_type}_{index}.wav"
        true_key = path_to_song + "y_true/" + true_key
        
        # get the wavelet transform for the training and true data
        Wavelets.getWaveletTransform(train_dict, key, level)
        Wavelets.getWaveletTransform(true_dict, true_key, level)

        # get the wavelet coefficients shapes
        if train_shape is None:
            train_shape = [c.shape for c in train.dwt]

        # get the wavelet coefficients
        true = true_dict[true_key]
        train_tensor = train.tensor_coeffs
        true_tensor = true.tensor_coeffs

        
        if np.all(train_tensor[0:] == 0) or np.all(true_tensor[0:] == 0) or np.all(train_tensor[1:] == 0) or np.all(true_tensor[1:] == 0): 
            continue  # skip this sample
        print(f"key: {key}")
        print(f"true_key: {true_key}")
        # add to the list
        y_train.append(train_tensor)
        y_true.append(true_tensor)

    return y_train, y_true, train_shape 

def batch_wavelets(path_to_training: str, stem_type: str, level: int =12, batch_size: int =8, max_songs: int =2, max_samples_per_song: int =10, num_features: int = 65536) -> tf.data.Dataset:
    '''
    Batch the wavelet data for training

    params:
    - path_to_training: str, path to the training data
    - stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    - level: int, level of wavelet decomposition
    - batch_size: int, size of the batch
    - max_songs: int, maximum number of songs to use
    - max_samples_per_song: int, maximum number of samples per song
    
    return: 
    - tf.data.Dataset, batched wavelet data
    '''
    

    # find all songs in the training data
    songs = os.listdir(path_to_training + stem_type)

    # limit the number of songs -- choose randomly
    if len(songs) > max_songs:
        songs = np.random.choice(songs, max_songs, replace=False)


    # generate pairs for each song
    y_train = []
    y_true = []
    shape = None
    for song in tqdm.tqdm(songs, desc=f"Generating Wavelet Batch: (level = {level}, batch_size = {batch_size}, max_songs = {max_songs}, max_samples_per_song = {max_samples_per_song})", total=len(songs), leave=True):
        # generate pairs for each song
        path_to_song = path_to_training + stem_type + '/' + song 
        if path_to_song[-1] != '/':
            path_to_song += '/'
        
        print(f"Path to song: {path_to_song}")

        ## check if the song is a directory
        if not os.path.isdir(path_to_song):
            continue

        ## check that song has y_train and y_true folders for the stem type
        if not os.path.isdir(path_to_song + 'y_train/') or not os.path.isdir(path_to_song + 'y_true/'):
            continue

        train, true, shape = generate_pairs(path_to_song, stem_type, level, max_songs_per_stem=max_samples_per_song)
        

        # add to the list
        y_train.append(train)
        y_true.append(true)

    # filter out empty samples or samples that have a mismatch in shape
    valid_indices = []


    for i in range(len(y_train)):
        if len(y_train[i]) == 0 or len(y_true[i]) == 0:
            continue
        if len(y_train[i].shape) != 3:
            continue
        valid_indices.append(i)

    y_train = [y_train[i] for i in valid_indices]
    y_true = [y_true[i] for i in valid_indices]


    y_train = np.concatenate(y_train)
    y_true = np.concatenate(y_true)
    # print(f"y_train shape: {y_train.shape}")
    # print(f"y_true shape: {y_train.shape}")
    # convert to tensors
    
    y_train = tf.convert_to_tensor(y_train)
    y_true = tf.convert_to_tensor(y_true)


    return y_train, y_true, shape
