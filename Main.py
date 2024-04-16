import Utils.Batch.generate_examples
import Utils.Batch.batch_data
import tensorflow as tf
import Models.wavelet_unet
import Config as cfg
from Train import train

### DO NOT CHANGE ###
MEDLEY2_PATH = 'Datasets/MedleyDB/V2/'
MEDLEY1_PATH = 'Datasets/MedleyDB/V1/'
TRAIN_PATH = 'Datasets/TrainingData/'

## Set current stem type to process. Options are: 'vocals', 'drums', 'bass', 'midrange'
CURR_STEM_TYPE = 'vocals'



def preprocess_medleydb(stem_type: str, clean: bool =False) -> None:
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
    Utils.Batch.generate_examples.generate_data(MEDLEY1_PATH, TRAIN_PATH, stem_type, 4) ## -- WORKS!
    Utils.Batch.generate_examples.generate_data(MEDLEY2_PATH, TRAIN_PATH, stem_type, 4) ## -- WORKS!


def batch_training_data(level: int = 12, batch_size: int = 8, max_songs: int = 2, max_samples_per_song: int = 10) -> tf.data.Dataset:
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
    dataset, shape = Utils.Batch.batch_data.batch_wavelets(TRAIN_PATH, 'vocals', level, batch_size, max_songs, max_samples_per_song)
    
    return dataset, shape

def main():

    model_config = cfg.cfg()

    ## Set the parameters -- might want to move to Config.py later
    WAVELET_DEPTH = model_config['wavelet_depth'] # level of wavelet decomposition
    BATCH_SIZE = model_config['batch_size'] # number of samples per batch
    MAX_SONGS = model_config['max_songs'] # maximum number of songs to include in the batch
    MAX_SAMPLES_PER_SONG = model_config['max_samples_per_song'] # maximum number of samples per song to include in the batch

    ## Set the batch parameters, pass to batch_training_data()
    BATCH_PARAMS = (WAVELET_DEPTH, BATCH_SIZE, MAX_SONGS, MAX_SAMPLES_PER_SONG)

    ## batch the data for medleyDB
    # preprocess_medleydb(CURR_STEM_TYPE, clean=False)

    ## set the batch size and epochs
    batch_size = model_config['batch_size']
    epochs = model_config['epochs']

    ## test that generate_pairs() works
    batched_training_data, shape = batch_training_data(*BATCH_PARAMS)

    
    
    ## define the model
    model = Models.wavelet_unet.WaveletUNet(model_config)

    # define a dummy input to build the model
    dummy_input = tf.random.normal(shape=(batch_size, WAVELET_DEPTH+1, model_config['num_coeffs'], model_config['channels']))
    
    # build the model
    model(dummy_input)

    # print the model summary
    model.summary()

    ## train the model
    model = train(model, batched_training_data, epochs, batch_size)


    
if __name__ == '__main__':
    main()