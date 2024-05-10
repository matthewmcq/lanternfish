# Lanternfish - Audio Source Separation Based on the Discrete Wavelet Transform

### Matthew McQuistion, Rayhan Meghji, Scott Petersen

## Introduction: 

Although the use of Neural Networks for audio source separation is not new, most commercially-available models are poor--both in separation quality, aliasing, and number of identified sources (usually limited to 4 stems). We wanted to leverage the use of the Discrete Wavelet Transform (DWT) to mitigate some of the issues faced by other models, which typically rely on the Short-Time Fourier Transform, which is sensitive to changes in phase and requires computationally-intensive post-processing. 

Furthermore, we feel that one of the largest drawbacks to other models is that they are intended to split an audio signal into a fixed number of sources all in one go. As faithful reconstruction of drums, vocals, and guitar information are fundamentally different, relying on one single model to handle sensitive high frequency details as well as precise rhythmic information simultaneously will always lead to trade-offs that negatively impact perceived audio quality. 

## Methodology:

The general architecture of our model is based around a U-Net architecture. To run the model, we trained on the Medley 1.0 and 2.0 datasets and permuted each example to expand the amount of audio we had access to. For preprocessing, we loaded in the mix and corresponding stem we were trying to isolate and computed the discrete wavelet transform to get the corresponding coefficients. Then, we converted each level of wavelet decomposition to the corresponding sub-band audio and fed that tensor into the U-Net. The model consisted of 12 layers of encoder/decoder blocks with a bottleneck between them and skip connections between corresponding blocks as well as with the summed input, which gives the actual audio. For the downsampling blocks, we used a size 15 filter with layer*24 output channels with leaky ReLu followed by a decimation layer that halved the number of timestep features between each block. After the bottleneck, we used a learnable upsampling layer to estimate and upsample the timesteps back to the original, concatenate the output with the corresponding skip connection and ended each block with a size 5 convolution. After the last decoding/upsampling block we concatenated the output with the summed input and used a size 1 tanh convolution to get the desired audio.

For the loss function, we used the output of the last layer (1 channel) and summed the three sub-bands of y_true to get the original audio output. Then we took the MSE. We did this so that the model would directly learn the correct audio, which minimizes the need for postprocessing and improves the model output.

To actually run the model on a full song, we first split the input 44.1kHz signal into 50% overlapping segments of length 32768 and resample them down to 22.05kHz samples of length 16384. Then, we run the model on all segments and use a Hanning window to seamlessly blend between the overlapping segments. This avoids boundary artifacts at the expense of additional runtime. Since we use separate models to separate each stem, we then subtract the output of each model from the original audio to make the separation easier for the next model. By the end, we have three separated stems: vocals, drums, and bass, and then subtract those three from the original to get the rest of the mix.

For training, we used a batch size of 16 with ~2000 iterations per epoch and a learning rate of 1-e4. We set the number of epochs arbitrarily high and used a callback function to train the model indefinitely until validation loss had not improved for 20 epochs, and then we would reset the model back to its best weights. Then we would run the model again with a batch size of 32 and a learning rate of 4-e5 for the same amount of iterations and callback value to refine it.

For vocals, we used 86 songs and allowed the model to look at 700 segments from the full set of permutations. For bass we had 84 songs and for drums we had 106. 
We also used l1 and l2 regularization on the kernel and activity/output of each layer. Our l1 value was 1-e12 and l2 was 1-e11.

## Reflection:

### How do you feel your project ultimately turned out? How did you do relative to your base/target/stretch goals?

We feel that our project turned out amazing. There is virtually no aliasing and it does a great job of splitting the stems with exceptionally high fidelity. That said, it came together at 2am on May 10th, which was definitely a little anxiety-inducing. We accomplished all of our base and target goals, but unfortunately ran out of time to accomplish our stretch goal.

### Did your model work out the way you expected it to?

Yes. It works very well and sounds similar to the outputs of similar models.

### How did your approach change over time? What kind of pivots did you make, if any? Would you have done differently if you could do your project over again?

For about a month our outputs were horrid and frankly disheartening despite the fact that loss was going down to very small amounts. Despite numerous changes to the architecture, the output remained suspiciously poor. However, when we ran the postprocessing directly on the trained model after only like 10 epochs (i.e. not loading it, instead doing it directly after model.fit) the output was clear as day. This really made us wonder if other iterations of our model worked…

At first, we tried to learn directly on the wavelet coefficients instead of learning on the actual audio. The initial drawback was that this required a lot more post processing and our interpolation scheme did not work super well with it. As a result, the model was actually really good at separating vocals from the rest of the mix, but the heavy aliasing meant that the output was not high quality enough to actually be useful. We decided to convert the split coefficients directly into the sub-band audios and train on those, but although there was less aliasing, the model was not as good at actually learning to separate the audio and there was still quite a bit of aliasing. Furthermore, when the model was tame enough to run locally, we used tf.image_resize with sinc interpolation, which worked really well, however when training on more data, we had to switch to Google CoLab, in which XLA did not support this operation. We spent about a week trying to manually implement this feature using convolution, but it didn't really work and we ended up going with a learnable interpolation layer from the Wave-U-Net paper. This, in part, also led us to move towards learning on audio, as that method was far better at learning continuous dependencies than the discrete coefficients.

### What do you think you can further improve on if you had more time?

There are a lot of very confusing small issues that have large impacts on model performance. If we had more time, we would like to try with another dataset and likely redo the data pipeline, as it is very bloated given the pivots we have had to make. Also, as these models take a long time and a lot of compute credits to train, it would be great if we had funding or simply more weeks to try different models and iterate that way. Also, we think it would be interesting to train on separating all stems at the same time rather than sequentially to test if our hypothesis on sequential separation being preferable was in fact correct. Lastly, we had a really cool idea of stem-agnostic separation, which currently does not exist, but unfortunately we ran out of time to work on this. 

### What are your biggest takeaways from this project/what did you learn?

Overall we learned a lot about digital signal processing and how different sample rates, interpolation schemes, and DSP techniques can affect audio quality. Designing a model around audio is difficult, because the small errors guaranteed in stochastic models are much more easily perceived by the human brain than formats like images or text. Managing incredibly fragile coefficients is difficult, and the size of each training example means that we really had to be clever about how we batched data and designed the pipeline to be as effective as possible. 

We also learned about XLA and how different computer architectures lend themselves better to certain types of architectures. This meant that we had to really think about the tradeoffs between output quality and training time/overall efficiency. 

There is so much more that goes into a large-scale project like this, and although we are glad that we took on such a large task, there is a lot more that goes into a deep learning project than simply arranging keras layers.

Also, by reading a lot of papers and combing through the associated GitHubs, we tried to make an effort to design our codebase in a way that mirrored theirs, which paid dividends by making the layouts super clear and allowed for flexibility when changing hyperparameters and tuning the model. 

[Google Drive link to models and Colab Backend](https://drive.google.com/drive/u/0/folders/1M_irkD7vvnZdminW1Haxmhs5-hbL09YB)

