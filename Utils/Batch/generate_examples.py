import numpy as np
import tensorflow as tf
import librosa
import os
import soundfile as sf
import numpy as np
import pywt
import cv2
from Wavelets import WaveData, SR

def split_audios(stem, duration=10):

    #Load in stems
    stem_audio, sr = librosa.load(stem, mono=False, sr=SR)

    # Transpose to have time as first dimension
    stem_audio = stem_audio.T

    # get total number of samples and number of chunks
    n_samples = len(stem_audio)
    print(n_samples)
    n_chunks = n_samples // (sr * duration)
    print(n_chunks)
    chunks = []

    # Split into chunks
    for i in range(n_chunks):
        start = i * sr * duration
        end = (i + 1) * sr * duration
        stem_chunk = stem_audio[start:end]
        chunks.append(stem_chunk)

    return chunks


def write_audio(chunks, stemname, destination):

    for i, chunk in enumerate(chunks):        
        sf.write(f'{destination}/{stemname}_{i}.wav', chunk, SR)


def split_per_folder(folder: str, PATH_DB: str, PATH_Train: str, stem_type: str, seconds: int =10) -> None:
    '''
    params:
    folder: str, name of the folder (song) in the db
    PATH_DB: str, path to the db
    PATH_Train: str, path to the training data
    stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    seconds: int, number of seconds to split the audio into
    '''
    # PATH_Train should be Datasets/TrainingData/
    # PATH_DB should be Datasets/DB/...

    # Get all stems in folder from db
    if not os.path.isdir(PATH_DB + folder):
        print(f"Folder {folder} does not exist. Skipping.")
        return None
    
    # if no folder for this stem type exists, create it
    if not os.path.isdir(PATH_Train + stem_type):
        os.makedirs(PATH_Train + stem_type)

    # if no folder for this specific song exists, create it
    if not os.path.isdir(PATH_Train + "/" + stem_type + "/" + folder):
        os.makedirs(PATH_Train + stem_type + "/" + folder)

        # make y_train and y_true folders
        os.makedirs(PATH_Train + stem_type + "/" + folder + "/y_train", exist_ok=True)
        os.makedirs(PATH_Train + stem_type + "/" + folder + "/y_true", exist_ok=True)

    TRAIN_PATH = PATH_Train + stem_type + "/" + folder + "/y_train"
    TRUE_PATH = PATH_Train + stem_type + "/" + folder + "/y_true"


    stem_folders = os.listdir(PATH_DB + folder)

    # get all folders that have the stem_type in their name (e.g. vocals_INSTR_SUM), should be only one
    desired_stem_dir = [folder + '/' + stem for stem in stem_folders if stem==f"{stem_type}_INSTR" or stem==f"{stem_type}_INSTR_SUM"]

    if len(desired_stem_dir) == 0:
        print("No desired stems found for this folder. Skipping.", folder)
        return None
    
    # find the true stem path in original db
    true_path_from_db = PATH_DB + desired_stem_dir[0] + f"/{stem_type}_INSTR_SUM.wav"
    
    # split true stem into chunks
    true_split = split_audios(true_path_from_db, seconds)
    
    ## write vox stem
    write_audio(true_split, stem_type, TRUE_PATH)

    # get all permutations that have the stem_type in their name
    perm_folders = [folder + '/' + stem_folder for stem_folder in stem_folders if "SUM" in stem_folder and f"{stem_type}" in stem_folder]
    
    for i, stem in enumerate(perm_folders):
        # find the perm stem path in original db
        perm_path_from_db = PATH_DB + stem + f"/{stem_type}_INSTR_SUM.wav"
        
        # split perm stem into chunks
        perm_split = split_audios(perm_path_from_db, seconds)

        ## write perm stem
        write_audio(perm_split, stem_type, TRAIN_PATH)


def split_all_folders(PATH_DB: str, PATH_Train: str, stem_type: str, seconds: int =10) -> None:
    '''
    params:
    PATH_DB: str, path to the db
    PATH_Train: str, path to the training data
    stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    seconds: int, number of seconds to split the audio into
    '''
    folders = os.listdir(PATH_DB)
    for folder in folders:
        split_per_folder(folder, PATH_DB, PATH_Train, stem_type, seconds)

def clean_training_data(PATH_Train: str, stem_type: str, folder: str=None) -> None:
    '''
    Cleans the training data set for a given stem type, removing all files in the y_true and y_train folders.

    params:
    PATH_Train: str, path to the training data
    stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    folder: str, name of the folder (song) in the db (OPTIONAL, if None, clean all folders in the training data)
    '''
    # get all folders in the training data
    folders = os.listdir(PATH_Train + '/' + stem_type)
    assert len(folders) > 0, "No folders found in the training data."

    # if only one folder is specified, only clean that folder
    if folder is not None:
        folders = [folder]
    
    # iterate over all folders
    for folder in folders:

        # get the paths to the y_true and y_train folders
        path_y_true = PATH_Train + '/' + stem_type + '/' + folder + '/y_true'
        path_y_train = PATH_Train + '/' + stem_type + '/' + folder + '/y_train'

        if not os.path.isdir(path_y_true):

            print(f"Folder {folder} does not exist. Skipping.")
            continue

        else:

            # get all files in the y_true and y_train folders
            y_true_files = os.listdir(path_y_true)
            y_train_files = os.listdir(path_y_train)

            # remove all files in the y_true and y_train folders
            for file in y_true_files:
                os.remove(path_y_true + '/' + file)
            for file in y_train_files:
                os.remove(path_y_train + '/' + file)