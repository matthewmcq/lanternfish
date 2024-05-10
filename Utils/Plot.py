import matplotlib.pyplot as plt
import numpy as np
import librosa
import Config as cfg

def display_waveforms(predict_true_0, predict_train_0, i):
    model_config = cfg.cfg()
    # Prepare the time axis
    time = np.arange(sum([len(coef) for coef in predict_true_0]))

    plt.rcParams['figure.figsize'] = (8, 8)
    plt.rcParams['font.size'] = 8

    # Plot the wavelet coefficients
    fig, ax = plt.subplots(model_config['wavelet_depth'] + 1, 1, figsize=(12, 6))

    # Plot the original signal
    start = 0
    for level, coef in enumerate(predict_true_0):
        ax[0].plot(time[start:start+len(coef)], coef, label=f'Level {level+1}')
        start += len(coef)
    ax[0].set_title('Original Signal')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()

    # Plot the wavelet coefficients
    start = 0
    for level in range(model_config['wavelet_depth']):
        ax[level + 1].plot(time[start:start+len(predict_true_0[-level])], predict_true_0[-level], label='True')
        ax[level + 1].plot(time[start:start+len(predict_train_0[-level])], predict_train_0[-level], label='Predicted')
        ax[level + 1].set_title(f'Wavelet Level {level + 1}')
        ax[level + 1].set_xlabel('Time')
        ax[level + 1].set_ylabel(f'Coef Level {level + 1}')
        ax[level + 1].legend()
        start += len(predict_true_0[-level])

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'audiofiles/sample{i}/waveforms.png')
    plt.close(fig)
    

def display_stft(coeffs, i, name):
    # Display the STFT of the predict_train_a3 audio
    fig, ax = plt.subplots(figsize=(6, 6))
    stft = librosa.stft(coeffs, n_fft=2048, hop_length=256)
    ax.pcolormesh(librosa.amplitude_to_db(np.abs(stft), ref=np.max), cmap='viridis')
    ax.set_title(f'STFT of predict_train_{name}, sample{i}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.savefig(f'audiofiles/sample{i}/{name}_stft.png')
    plt.close(fig)
