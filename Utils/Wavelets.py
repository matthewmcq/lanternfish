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
    coeffs_left = pywt.wavedec(data[song].waveform[:, 0], 'db1', level=level)
    coeffs_right = pywt.wavedec(data[song].waveform[:, 1], 'db1', level=level)
    # print(f"Left coefficients shape: {[c.shape for c in coeffs_left]}")

    # Find the maximum length among all coefficients
    max_len = max([c.shape[0] for c in coeffs_left + coeffs_right])

    # Stretch the coefficients to the maximum length using interpolation
    stretched_coeffs_left = []
    stretched_coeffs_right = []
    for c_left, c_right in zip(coeffs_left, coeffs_right):
        stretched_left = cv2.resize(c_left.reshape(1, -1), (max_len, 1), interpolation=cv2.INTER_NEAREST).flatten()
        stretched_right = cv2.resize(c_right.reshape(1, -1), (max_len, 1), interpolation=cv2.INTER_NEAREST).flatten()
        stretched_coeffs_left.append(stretched_left)
        stretched_coeffs_right.append(stretched_right)

    # Stack the stretched coefficients along the channel axis
    stacked_coeffs = np.stack([stretched_coeffs_left, stretched_coeffs_right], axis=-1)

    # Convert the stacked coefficients to a TensorFlow tensor
    tensor_coeffs = tf.convert_to_tensor(stacked_coeffs)
    # print(f"Tensor coefficients shape: {tensor_coeffs.shape}")

    # Update the data object
    data[song].dwt = coeffs_left
    data[song].tensor_coeffs = tensor_coeffs

    return data

def makeWaveDict(folder_name: str) -> dict:
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