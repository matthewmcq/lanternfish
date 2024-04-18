import matplotlib.pyplot as plt
import numpy as np


def visualize_wavelet(coeffs, sr=44100, duration=None):
    levels = coeffs.shape[0]
    if duration is None:
        duration = coeffs.shape[1] / sr

    fig, ax = plt.subplots(figsize=(50, 15))
    for i in range(levels):
        ci = np.abs(coeffs[i])
        ax.imshow(np.sqrt(ci.reshape(1, -1)), cmap='inferno', aspect='auto',
                  extent=[0, duration, i, i + 1], interpolation='nearest')

    ax.set_title("Wavelet Coefficients")
    ax.set_ylabel("Level")
    level_labels = [f"Level {levels - i}" for i in range(levels)]
    ax.set_yticks(np.arange(0.5, levels, 1))
    ax.set_yticklabels(level_labels)
    ax.set_xlim([0, duration])
    ax.set_ylim([0, levels])
    ax.set_xlabel("Time (s)")
    plt.colorbar(ax.imshow(coeffs[0].reshape(1, -1), aspect='auto'), ax=ax, label="Coefficient Magnitude")
    plt.tight_layout()
    plt.show()