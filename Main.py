import Utils
import os

import Utils.Batch
import Utils.Batch.generate_examples
import Utils.Batch.batch_data

MEDLEY_PATH = 'Datasets/MedleyDB/V2/'
TRAIN_PATH = 'Datasets/TrainingData/'

def main():

    ## batch the data for medleyDB


    ## call generate_examples() to generate the examples
    Utils.Batch.generate_examples.generate_data(MEDLEY_PATH, TRAIN_PATH, 'vocals', 10) ## -- WORKS!

    ## call clean_training_data() to clean the training data if something goes wrong
    # Utils.Batch.generate_examples.clean_training_data(TRAIN_PATH, 'vocals')

    ## test that generate_pairs() works
    dataset = Utils.Batch.batch_data.batch_wavelets(TRAIN_PATH, 'vocals', 4, 4, 2, 2)
    print(f"Dataset: {dataset.element_spec}")
    
    

if __name__ == '__main__':
    main()