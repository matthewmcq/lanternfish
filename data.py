import numpy as np

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
