import tensorflow as tf
import argparse
import numpy as np

# Use python Evaluate/audio_metrics.py reference_filepath estimate_filepath

# python Evaluate/audio_metrics.py Evaluate/TestData/true.wav Evaluate/TestData/train.wav

def read_wav(file_path):
    audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(file_path), desired_channels=1)
    audio = tf.cast(audio, tf.float32) / np.iinfo(np.int16).max
    return audio, sample_rate

def calculate_sdr(reference, estimate, non_zero_mask):
    reference = tf.boolean_mask(reference, non_zero_mask)
    estimate = tf.boolean_mask(estimate, non_zero_mask)
    num = tf.reduce_sum(reference * estimate, axis=-1)
    denom = tf.reduce_sum(reference ** 2, axis=-1)
    sdr = 10 * tf.math.log(tf.reduce_mean(num / denom))
    return sdr

def calculate_sar(reference, estimate, non_zero_mask):
    reference = tf.boolean_mask(reference, non_zero_mask)
    estimate = tf.boolean_mask(estimate, non_zero_mask)
    num = tf.reduce_sum(reference ** 2, axis=-1)
    denom = tf.reduce_sum((reference - estimate) ** 2, axis=-1)
    sar = 10 * tf.math.log(tf.reduce_mean(num / denom))
    return sar

def calculate_sir(reference, estimate, non_zero_mask):
    reference = tf.boolean_mask(reference, non_zero_mask)
    estimate = tf.boolean_mask(estimate, non_zero_mask)
    num = tf.reduce_sum(reference ** 2, axis=-1)
    denom = tf.reduce_sum((estimate - reference) ** 2, axis=-1)
    sir = 10 * tf.math.log(tf.reduce_mean(num / denom))
    return sir

def evaluate(reference_file, estimate_file):
    reference, _ = read_wav(reference_file)
    estimate, _ = read_wav(estimate_file)

    non_zero_mask = tf.math.not_equal(reference, 0.0) & tf.math.not_equal(estimate, 0.0)

    sdr = calculate_sdr(reference, estimate, non_zero_mask)
    sar = calculate_sar(reference, estimate, non_zero_mask)
    sir = calculate_sir(reference, estimate, non_zero_mask)

    return sdr, sar, sir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio separation/enhancement")
    parser.add_argument("reference_file", type=str, help="Path to the reference audio file")
    parser.add_argument("estimate_file", type=str, help="Path to the estimated audio file")
    args = parser.parse_args()

    sdr, sar, sir = evaluate(args.reference_file, args.estimate_file)
    print(f"SDR: {sdr.numpy():.2f} dB")
    print(f"SAR: {sar.numpy():.2f} dB")
    print(f"SIR: {sir.numpy():.2f} dB")