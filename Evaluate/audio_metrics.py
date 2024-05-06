import tensorflow as tf
import argparse
import numpy as np

# Use python Evaluate/audio_metrics.py reference_filepath estimate_filepath
# python Evaluate/audio_metrics.py Evaluate/TestData/true.wav Evaluate/TestData/train.wav

# Results:
# SDR: -5.05 dB
# SI-SAR: -5.05 dB
# SI-SIR: 38.03 dB
# SNDR: 0.07 dB

def read_wav(file_path):
    audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(file_path), desired_channels=1)
    audio = tf.cast(audio, tf.float32) / np.iinfo(np.int16).max
    return audio, sample_rate

# def calculate_sdr(reference, estimate, non_zero_mask):
#     reference = tf.boolean_mask(reference, non_zero_mask)
#     estimate = tf.boolean_mask(estimate, non_zero_mask)
#     num = tf.reduce_sum(reference * estimate, axis=-1)
#     denom = tf.reduce_sum(reference ** 2, axis=-1)
#     sdr = 10 * tf.math.log(tf.reduce_mean(num / denom))
#     return sdr

def calculate_si_sdr(reference, estimate, non_zero_mask):
    reference = tf.boolean_mask(reference, non_zero_mask)
    estimate = tf.boolean_mask(estimate, non_zero_mask)
    num = tf.reduce_sum(reference ** 2, axis=-1)
    denom = tf.reduce_sum((estimate - reference) ** 2, axis=-1)
    si_sdr = 10 * tf.math.log(tf.reduce_mean(num / denom))
    return si_sdr

def calculate_si_sar(reference, estimate, non_zero_mask):
    reference = tf.boolean_mask(reference, non_zero_mask)
    estimate = tf.boolean_mask(estimate, non_zero_mask)
    etarget = reference
    eres = estimate - reference
    einterf = tf.reduce_sum(eres * reference, axis=-1, keepdims=True) * reference / tf.reduce_sum(reference ** 2, axis=-1, keepdims=True)
    eartif = eres - einterf
    num = tf.abs(tf.reduce_sum(etarget ** 2, axis=-1))
    denom = tf.abs(tf.reduce_sum(eartif ** 2, axis=-1))
    si_sar = 10 * tf.math.log(tf.reduce_mean(num / denom))
    return si_sar

def calculate_si_sir(reference, estimate, non_zero_mask):
    reference = tf.boolean_mask(reference, non_zero_mask)
    estimate = tf.boolean_mask(estimate, non_zero_mask)
    etarget = reference
    eres = estimate - reference
    einterf = tf.reduce_sum(eres * reference, axis=-1, keepdims=True) * reference / tf.reduce_sum(reference ** 2, axis=-1, keepdims=True)
    num = tf.abs(tf.reduce_sum(etarget ** 2, axis=-1))
    denom = tf.abs(tf.reduce_sum(einterf ** 2, axis=-1))
    si_sir = 10 * tf.math.log(tf.reduce_mean(num / denom))
    return si_sir

def calculate_sndr(reference, estimate, non_zero_mask):
    reference = tf.boolean_mask(reference, non_zero_mask)
    estimate = tf.boolean_mask(estimate, non_zero_mask)
    p = reference
    e = reference - estimate
    sigma_n = tf.math.reduce_std(estimate)
    p_max = tf.reduce_max(tf.abs(p))
    sigma_e = tf.math.reduce_std(e)
    num = p_max ** 2
    denom = sigma_e ** 2 + sigma_n ** 2
    sndr = 10 * tf.math.log(num / denom)
    return sndr

def evaluate(reference_file, estimate_file):
    reference, _ = read_wav(reference_file)
    estimate, _ = read_wav(estimate_file)
    non_zero_mask = tf.math.not_equal(reference, 0.0) & tf.math.not_equal(estimate, 0.0)
    sdr = calculate_si_sdr(reference, estimate, non_zero_mask)
    sar = calculate_si_sar(reference, estimate, non_zero_mask)
    sir = calculate_si_sir(reference, estimate, non_zero_mask)
    sndr = calculate_sndr(reference, estimate, non_zero_mask)
    return sdr, sar, sir, sndr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio separation/enhancement")
    parser.add_argument("reference_file", type=str, help="Path to the reference audio file")
    parser.add_argument("estimate_file", type=str, help="Path to the estimated audio file")
    args = parser.parse_args()
    sdr, sar, sir, sndr = evaluate(args.reference_file, args.estimate_file)
    print(f"SI-SDR: {sdr.numpy():.2f} dB")
    print(f"SI-SAR: {sar.numpy():.2f} dB")
    print(f"SI-SIR: {sir.numpy():.2f} dB")
    print(f"SNDR: {sndr.numpy():.2f} dB")

# import tensorflow as tf
# import argparse
# import numpy as np

# # Use python Evaluate/audio_metrics.py reference_filepath estimate_filepath
# # python Evaluate/audio_metrics.py Evaluate/TestData/true.wav Evaluate/TestData/train.wav

# def read_wav(file_path):
#     audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(file_path), desired_channels=1)
#     audio = tf.cast(audio, tf.float32) / np.iinfo(np.int16).max
#     return audio, sample_rate

# def calculate_sdr(reference, estimate, non_zero_mask):
#     reference = tf.boolean_mask(reference, non_zero_mask)
#     estimate = tf.boolean_mask(estimate, non_zero_mask)
#     num = tf.reduce_sum(tf.abs(reference), axis=-1)
#     denom = tf.reduce_sum(tf.abs(estimate - reference), axis=-1)
#     eps = 1e-8
#     sdr = 10 * tf.math.log((num + eps) / (denom + eps))
#     return sdr

# def calculate_si_sar(reference, estimate, non_zero_mask):
#     reference = tf.boolean_mask(reference, non_zero_mask)
#     estimate = tf.boolean_mask(estimate, non_zero_mask)
#     etarget = reference
#     eres = estimate - reference
#     einterf = tf.reduce_sum(eres * reference, axis=-1, keepdims=True) * reference / (tf.reduce_sum(reference ** 2, axis=-1, keepdims=True) + 1e-8)
#     eartif = eres - einterf
#     num = tf.reduce_sum(tf.abs(etarget), axis=-1)
#     denom = tf.reduce_sum(tf.abs(eartif), axis=-1)
#     eps = 1e-8 
#     si_sar = 10 * tf.math.log((num + eps) / (denom + eps))
#     return si_sar

# def calculate_si_sir(reference, estimate, non_zero_mask):
#     reference = tf.boolean_mask(reference, non_zero_mask)
#     estimate = tf.boolean_mask(estimate, non_zero_mask)
#     etarget = reference
#     eres = estimate - reference
#     einterf = tf.reduce_sum(eres * reference, axis=-1, keepdims=True) * reference / (tf.reduce_sum(reference ** 2, axis=-1, keepdims=True) + 1e-8)
#     num = tf.reduce_sum(tf.abs(etarget), axis=-1)
#     denom = tf.reduce_sum(tf.abs(einterf), axis=-1)
#     eps = 1e-8  
#     si_sir = 10 * tf.math.log((num + eps) / (denom + eps))
#     return si_sir

# def calculate_sndr(reference, estimate, non_zero_mask):
#     reference = tf.boolean_mask(reference, non_zero_mask)
#     estimate = tf.boolean_mask(estimate, non_zero_mask)
#     p = reference
#     e = reference - estimate
#     sigma_n = tf.math.reduce_std(estimate)
#     p_max = tf.reduce_max(tf.abs(p))
#     sigma_e = tf.math.reduce_std(e)
#     num = p_max ** 2
#     denom = sigma_e ** 2 + sigma_n ** 2
#     eps = 1e-8 
#     sndr = 10 * tf.math.log((num + eps) / (denom + eps))
#     return sndr

# def evaluate(reference_file, estimate_file):
#     reference, _ = read_wav(reference_file)
#     estimate, _ = read_wav(estimate_file)
#     non_zero_mask = tf.math.not_equal(reference, 0.0) & tf.math.not_equal(estimate, 0.0)
#     sdr = calculate_sdr(reference, estimate, non_zero_mask)
#     si_sar = calculate_si_sar(reference, estimate, non_zero_mask)
#     si_sir = calculate_si_sir(reference, estimate, non_zero_mask)
#     sndr = calculate_sndr(reference, estimate, non_zero_mask)
#     return sdr, si_sar, si_sir, sndr

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate audio separation/enhancement")
#     parser.add_argument("reference_file", type=str, help="Path to the reference audio file")
#     parser.add_argument("estimate_file", type=str, help="Path to the estimated audio file")
#     args = parser.parse_args()
#     sdr, si_sar, si_sir, sndr = evaluate(args.reference_file, args.estimate_file)
#     print(f"SDR: {sdr.numpy():.2f} dB")
#     print(f"SI-SAR: {si_sar.numpy():.2f} dB")
#     print(f"SI-SIR: {si_sir.numpy():.2f} dB")
#     print(f"SNDR: {sndr.numpy():.2f} dB")