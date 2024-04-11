## CSCI 1470 Final Project -- Wavelet-based Deep Learning Stem Splitter

## Who: Matthew McQuistion, Rayhan Meghji, Scott Petersen

## Introduction: What problem are you trying to solve and why?

Although the use of Neural Networks for audio source separation is not new, most commercially-available models are poor--both in separation quality, aliasing, and number of identified sources (usually limited to 4 stems). We wanted to leverage the use of the Discrete Wavelet Transform (DWT) to mitigate some of the issues faced by other models, which typically rely on the Short-Time Fourier Transform, which is sensitive to changes in phase and requires computationally-intensive post-processing. 

Furthermore, we feel that one of the largest drawbacks to other models is that they are intended to split an audio signal into a fixed number of sources all in one go. As faithful reconstruction of drums, vocals, and guitar information are fundamentally different, relying on one single model to handle sensitive high frequency details as well as precise rhythmic information simultaneously will always lead to trade-offs that negatively impact perceived audio quality. 

## Related Work: Are you aware of any, or is there any prior work that you drew on to do your project?

Yes, we are continuing work on developing the "Wave-UNet" architecture proposed in the paper "Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation" authored by Daniel Stoller, Sebastian Ewert, and Simon Dixon. In essence, this source separation architecture learns to separate STFT magnitudes and reconstruct the sources with the Griffin-Lim algorithm. This model's advantage over previous approaches is the use of learnable interpolation layers in the upsampling (decoding) blocks to reduce aliasing and noise from decimated features. 
Our model differs from the "Wave-UNet" in two key ways: 1) We use DWT scalograms instead of STFT spectrograms, which eliminates the risk of phase issues and the need for the GL algorithm, which is slow and computationally intensive; 2) the "Wave-UNet" is trained to predict K sources from an audio signal all in one go, while each block of our model either isolates one source or separates an arbitrary number of mid-range instruments.
[link](https://arxiv.org/abs/1806.03185)

Another paper we found extremely helpful is "Time-Domain Audio Source Separation Based on Wave-U-Net Combined with Discrete Wavelet Transform" authored by Tomohiko Nakamura and Hiroshi Saruwatari. We actually reached out to Tomohiko Nakamura via email, and he was kind enough to provide us with the GitHub repository used for the paper, which we cannot understate our gratitude for. This paper   uses the waveform of the audio signal instead of the STFT, and each DS/US block implements its own DWT/IDWT through a lifting scheme with learnable wavelet filters for perfect reconstruction.
We might need to implement this ourselves, but using this model in conjunction with the "Wave-UNet" for sources that require a wider frequency range and/or more timbral distinctions for analysis (e.g. drums, or the final pass of midrange stems). That said, this approach is a bit "overkill" for what we are going for, and our idea is to take some conceptual aspects of this DWT-based model and simplify them to work with a slightly modified version of the original Wave-UNet setup.
[link](https://arxiv.org/abs/2001.10190)

## Data:
We are using the Medley2.0 Database as well as MoisesDB, which are two large datasets for audio source separation.

## Methodology: What is the architecture of your model?

We are using a UNet Style architecture that is a modified version of the "Wave-UNet" as well as the "MRA Wave-UNet." We are training it primarily on a CoLab Pro+ instance with 86GB RAM and an NVIDIA A100 GPU with 40GB VRAM.

The beauty of the setup we are going for is that our MVP is just a vocal separator, and we can continually iterate and add more stem separation support to the framework.

Our solution is as follows: we propose using multiple separately trained models that are each tailored to separate a specific instrument, and run them in series using phase-cancellation on the reconstructed waveform signal to progressively separate audio sources while "stripping" audio information from the original signal to reduce complexity for subsequent separator models.

## Metrics: What constitutes “success?”

Given the nature of audio signals, perceptual metrics are going to be critical. In addition to using our own ears, we can make use of common industry metrics like Signal to Distortion Ratio (SDR) for each stem to individually tune each model to perform better for its given instrument.

**Target Goal:** Full source separation (vocals, drums, bass, and all mid-range instruments)

**Base Goal:** Vocal Isolation

**Stretch Goal:** Arbitrary stem separation for the midrange instruments (i.e. have acoustic guitar, electric guitar, violin, etc. all separated into individual stems)

## Ethics:

- What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?

Using the MoisesDB and Medley2.0 datasets means we are primarily training on Western music (typically classical, rock, and pop), which means that not only will not model likely struggle to generalize to other styles of music, the structure of the series model architecture might implicitly reflect our own biases for which stems are commonly found in music. That said, if we can accomplish arbitrary stem number separation with a quantitative criteria to separate based purely on timbre (e.g. MFCC or Fourier Analysis), this should mitigate some of the bias.

- Why is Deep Learning a good approach to this problem?

Not only is there no way to perfectly split stems analytically, humans have a naturally good intuition for what sources make up an audio signal like a song. As such, using a DL framework that learns based off similar criteria to the way humans distinguish between audio sources is not only a natural approach to tackling source separation, but one that currently cannot be done any other way.

## Division of Labor:

**Matthew McQuistion** - Wavelet Analysis and generation of training examples, MRA implementation, pre/post processing

**Rayhan Meghji** - Preprocessing Medley2.0, Wave-UNet Implementation

**Scott Petersen** - Preprocessing MoisesDB, Wave-UNet Implementation
