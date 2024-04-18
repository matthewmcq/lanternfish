import matplotlib.pyplot as plt
import numpy as np



def visualize_wavelet(coeffs):
    levels = coeffs.shape[0]
    channels = 1
    coeffs = coeffs.numpy()  # Convert TensorFlow tensor to NumPy array
    duration = 4  # Duration of the audio snippet in seconds

    fig, ax = plt.subplots(figsize=(50, 15))
    for i in range(levels):
    
        ci = np.abs(coeffs[i, :])
        # max_val = np.max(ci)
        # if max_val == 0:
        #     max_val = 1
        # ci /= max_val
        # vmin = np.min(ci)
        # vmax = np.max(ci)
        ax.imshow(np.sqrt(ci.reshape(1, -1)), cmap='inferno',  aspect='auto',
                extent=[0, duration, i * channels, i * channels + 1], interpolation='nearest')

    ax.set_title("Wavelet Coefficients")
    ax.set_ylabel("Level and Channel")

    level_labels = [f"Level {levels - i}" for i in range(levels)]
    channel_labels = [f"Channel {ch + 1}" for ch in range(channels)]
    level_channel_labels = [f"{level} {channel}" for level in level_labels for channel in channel_labels]

    ax.set_yticks(np.arange(0.5, levels * channels, 1))
    ax.set_yticklabels(level_channel_labels)
    ax.set_xlim([0, duration])
    ax.set_ylim([0, levels * channels])
    ax.set_xlabel("Time (s)")

    plt.colorbar(ax.imshow(coeffs[0, :].reshape(1, -1), aspect='auto'), ax=ax, label="Coefficient Magnitude")
    plt.tight_layout()
    plt.show()