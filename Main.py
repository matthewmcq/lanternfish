import Utils.Batch.generate_examples
import Utils.Batch.batch_data
import Utils.Plot
import tensorflow as tf
import Models.wavelet_unet
import Config as cfg
from Train import train, WaveletLoss
import pywt
import soundfile as sf
import numpy as np
from Utils.Wavelets import inverseWaveletReshape

### DO NOT CHANGE ###
MEDLEY2_PATH = 'Datasets/MedleyDB/V2/'
MEDLEY1_PATH = 'Datasets/MedleyDB/V1/'
TRAIN_PATH = 'Datasets/TrainingData/'

## Set current stem type to process. Options are: 'vocals', 'drums', 'bass', 'midrange'
CURR_STEM_TYPE = 'vocals'

SR = 44100

def preprocess_medleydb(stem_type: str, clean: bool =False, sample_length=65536) -> None:
    '''
    Preprocess the MedleyDB dataset to generate training data

    params:
    - stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    - clean: bool, flag to clean the training data

    return: None
    '''

    ## call clean_training_data() first to clean the training data if something goes wrong
    if clean: 
        Utils.Batch.generate_examples.clean_training_data(TRAIN_PATH, stem_type)

    ## call generate_examples() to generate the examples
    Utils.Batch.generate_examples.generate_data(MEDLEY1_PATH, TRAIN_PATH, stem_type, sample_length) ## -- WORKS!
    Utils.Batch.generate_examples.generate_data(MEDLEY2_PATH, TRAIN_PATH, stem_type, sample_length) ## -- WORKS!


def batch_training_data(level: int = 12, batch_size: int = 8, max_songs: int = 2, max_samples_per_song: int = 10, num_features: int=65536) -> tf.data.Dataset:
    '''
    Batch the wavelet data for training

    params:
    - level: int, level of wavelet decomposition
    - batch_size: int, number of samples per batch
    - max_songs: int, maximum number of songs to include in the batch
    - max_samples_per_song: int, maximum number of samples per song

    return: 
    - tf.data.Dataset, batched wavelet data
    '''
    ## call batch_wavelets() to batch the wavelet data
    y_train, y_true, shape = Utils.Batch.batch_data.batch_wavelets(TRAIN_PATH, 'vocals', level, batch_size, max_songs, max_samples_per_song, num_features)
    
    return y_train, y_true, shape

def main():

    # model_config = cfg.cfg()
    model_config = cfg.cfg()

    ## Set the parameters -- might want to move to Config.py later
    WAVELET_DEPTH = model_config['wavelet_depth'] # level of wavelet decomposition
    BATCH_SIZE = model_config['batch_size'] # number of samples per batch
    MAX_SONGS = model_config['max_songs'] # maximum number of songs to include in the batch
    MAX_SAMPLES_PER_SONG = model_config['max_samples_per_song'] # maximum number of samples per song to include in the batch

    ## Set the batch parameters, pass to batch_training_data()
    BATCH_PARAMS = (WAVELET_DEPTH, BATCH_SIZE, MAX_SONGS, MAX_SAMPLES_PER_SONG)

    ## batch the data for medleyDB
    # preprocess_medleydb(CURR_STEM_TYPE, clean=True)
    

    ## set the batch size and epochs
    batch_size = model_config['batch_size']
    epochs = model_config['epochs']

    ## test that generate_pairs() works
    y_train, y_true, shape = batch_training_data(*BATCH_PARAMS)
    print(f"shape at beginning of main: {shape}")
    # print(shape.type)
    # print([s.type for s in shape])

    # (num_samples, num_coeffs, wavelet_depth+1) tf.Tensors
    # (3, num_coeffs, wavelet_depth+1)

    print(f"y_train[0].shape: {y_train[0].shape}")

    # Utils.Plot.visualize_wavelet(y_train[0])

    # exit()
    
    # print(y_train)
    # print(y_true)
        
    # ## define the model
    # model = Models.wavelet_unet.WaveletUNet(
    #     num_coeffs=model_config['num_coeffs'],
    #     wavelet_depth=model_config['wavelet_depth'],
    #     batch_size=model_config['batch_size'],
    #     channels=model_config['channels'],
    #     num_layers=model_config['num_layers'],
    #     num_init_filters=model_config['num_init_filters'],
    #     filter_size=model_config['filter_size'],
    #     l1_reg=model_config['l1_reg'],
    #     l2_reg=model_config['l2_reg']
    #     )

    # # define a dummy input to build the model
    # model(tf.random.normal(shape=(batch_size, model_config['num_coeffs'], WAVELET_DEPTH+1)))

    # # print the model summary
    # model.summary()

    ## check the loss function for all zeros
    zero_train = tf.zeros_like(y_train)


    wavelet_loss = WaveletLoss(
        wavelet_level=model_config['wavelet_depth'],
        lambda_vec=model_config['lambda_vec'],
        lambda_11=model_config['lambda_11'],
        lambda_12=model_config['lambda_12'],
    )

    # ## check default loss:
    loss = WaveletLoss( wavelet_level=4, lambda_vec=[40, 2.5, 0.3, 0.2], lambda_11=1, lambda_12=0.25, name='wavelet_loss')
    # print("Default Loss with regularization:", loss(y_true, y_train))
    # print("Default Loss (All zeros):", loss(y_true, zero_train))


    ## train the model
    # model = train(model, wavelet_loss, y_train, y_true, epochs, batch_size)

    # # use nif30 (?)
    # model.save('wavelet_unet_model_nif30.keras')

    loaded_model = tf.keras.models.load_model('wavelet_unet_model_nif27_filter11.keras')
    loaded_model(tf.random.normal(shape=(batch_size, model_config['num_coeffs'], WAVELET_DEPTH+1)))
    # loaded_model = tf.keras.models.load_model('wavelet_unet_model.h5')

    # loaded_model.summary()

    # print(f"y_train[0].shape: {y_train[0].shape}")

    # predict_train_0 = loaded_model.predict(tf.expand_dims(y_train[0], axis=0))

    # print(f"predict_train_0: {predict_train_0}")
    # print(f"predict_train_0.shape: {predict_train_0.shape}")

    # print(f"y_true[0]: {y_true[0]}")
    # print(f"y_true[0].shape: {y_true[0].shape}")

    # predict_train_0 = inverseWaveletReshape(predict_train_0, shape, model_config['wavelet_depth'])

    # predict_train_0 = list(predict_train_0)
    # # print(f"predict_train_0.shape: {predict_train_0}")
    # # print(f"predict_train_0: {predict_train_0}")
    # output = pywt.waverec(predict_train_0, 'haar')
    # print(f"output: {output}")
    # print(f"output.shape: {output.shape}")
    # sf.write('test.wav', output, SR)
    # sf.write('true.wav', y_true[0], SR)

    for i in range(10):
        predict_train_0 = loaded_model.predict(tf.expand_dims(y_train[i], axis=0))[0]
        # predict_true_0 = tf.expand_dims(y_true[i], axis=0).numpy()
        predict_true_0 = y_true[i]
        print(f"y_true_i.shape: {y_true[i].shape}")
        print(f"y_train_i.shape: {predict_train_0.shape}")
        # print(f"predict_train_0.shape: {predict_train_0.shape}")
        predict_true_0 = inverseWaveletReshape(predict_train_0, shape, model_config['wavelet_depth'])
        print(f"predict_true_0: {predict_true_0}")
        print(f"y_true_i: { tf.expand_dims(y_true[i], axis=0).numpy()}")
        print([c.shape for c in predict_true_0])
        # predict_true_0 = list(predict_true_0)
        # print(f"predict_train_0.shape no IWTReshape: {(predict_train_0)}")
        # output = pywt.waverec(predict_true_0, 'haar')
        output = pywt.upcoef('a', predict_true_0[0], 'haar', level=model_config['wavelet_depth'])
        print(f"output in loop: {output}")
        print(f"output.shape in loop: {output.shape}")
        sf.write(f'test_upcoef{i}.wav', output, SR)
        # sf.write(f'true{i}.wav', y_true[i], SR//2)

    
if __name__ == '__main__':
    main()