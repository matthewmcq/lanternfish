import numpy as np
import tensorflow as tf
import librosa
import cv2
import tqdm
import pywt
import os


## DO NOT CHANGE
SR = 44100

class WaveData:
    def __init__(self, 
                 filename: str, 
                 waveform: np.ndarray, 
                 dwt: np.ndarray) -> None:
        
        self.filename = filename
        self.waveform = waveform
        self.dwt = dwt
        self.tensor_coeffs = None

    filename = "UNINITIALIZED"
    waveform = None
    dwt = None
    tensor_coeffs = None


def getWaveletTransform(data: dict, song: str, level: int=12) -> dict:
    '''
    Get the wavelet transform of the waveform

    params:
    - data: dict, dictionary of wavelet data
    - song: str, song key
    - level: int, level of wavelet decomposition

    return: 
    - data: dict, updated dictionary of wavelet data
    '''

    # ensure the waveform is in the correct shape
    if data[song].waveform.shape[0] == 2:
        data[song].waveform = np.transpose(data[song].waveform)
    # print(f"Left channel waveform: {data[song].waveform[:, 0]}")

    # Perform wavelet decomposition
    coeffs_left = pywt.wavedec(data[song].waveform[:, 0], 'haar', level=level, mode='reflect')

    
    # Find the maximum length among all coefficients
    # max_len = max([c.shape[0] for c in coeffs_left + coeffs_right])
    max_len = max([c.shape[0] for c in coeffs_left])

    # print([c.shape[0] for c in coeffs_left])

    # Stretch the coefficients to the maximum length using interpolation
    stretched_coeffs_left = []
    # stretched_coeffs_right = []
    for c_left in coeffs_left:
        stretched_left = cv2.resize(c_left.reshape(1, -1), (max_len, 1), interpolation=cv2.INTER_NEAREST).flatten()
        stretched_coeffs_left.append(stretched_left)

    ## transpose
    stretched_coeffs_left = np.transpose(stretched_coeffs_left)
    tensor_coeffs = tf.convert_to_tensor(stretched_coeffs_left)

    # Update the data object
    data[song].dwt = coeffs_left
    data[song].tensor_coeffs = tensor_coeffs

    return data

def makeWaveDict(folder_name: str, indices: list) -> dict:
    '''
    Make a dictionary of wavelet data

    params:
    - folder_name: str, name of the folder containing the waveforms

    return: 
    - data: dict, dictionary of wavelet data
    '''

    data = {}

    # Get all the filenames in the folder
    filenames = os.listdir(folder_name)
    filenames = [folder_name + filename for filename in filenames if filename.endswith('.wav')]

    if indices is not None:
        filenames = [filenames[i] for i in indices]

    song_name = folder_name.split('/')[-2]

    for filename in tqdm.tqdm(filenames, desc=f'Loading waveforms for {song_name}', total=len(filenames), leave=False):
        # print(f"Loading {filename}")

        # Load the waveform
        waveform, _ =librosa.load(filename, sr=SR, mono=False)
        # print(f"Waveform shape: {waveform.shape}")

        # Transpose to have time as first dimension
        np.transpose(waveform)

        # Create a WaveData object
        data[filename] = WaveData(filename, waveform, None)
    return data


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
        level_coeffs = coeffs[:, level].numpy()
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
