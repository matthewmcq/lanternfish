import librosa
import os
import soundfile as sf
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Wavelets import SR

def add_full_audio_to_perms(folder: str, PATH_DB: str) -> None:
    '''
    Adds the full audio file to the permutations of the stems in the folder.

    params:
    - folder: str, name of the folder (song) in the db
    - PATH_DB: str, path to the db

    return: None
    '''

    # get the path to the folder and the song name
    song_name = folder
    song_path = PATH_DB + song_name

    # check if the folder exists
    if not os.path.isdir(song_path):
        print(f"Folder {song_name} does not exist. Skipping.")
        return None
    
    # get the stems in the folder
    stems = os.listdir(song_path)
    stems = [stem for stem in stems if stem.endswith('_STEMS')]

    # check if the folder has a stem folder
    assert len(stems) == 1, "More than one stem folder found."

    # get the stem names and add them to the song name to create the folder name
    stems = stems[0]
    stems_path = song_path + '/' + stems
    stem_names = os.listdir(stems_path)
    stem_names = [stem for stem in stem_names if stem.endswith('.wav')] # list of all stems in the folder that are .wav files

    for stem in stem_names:
        song_name += f"_{stem.split('.')[0]}" # remove the .wav extension

    ## copy the full audio to the folder
    full_audio = PATH_DB + folder + '/' + folder + '_MIX.wav'

    # Reduce size of song name by eliminating ann "_SUM" and + "_INSTR" and then adding one of each of them back
    song_name = song_name.replace("_SUM", "").replace("_INSTR", "")
    song_name += "_INSTR"
    song_name += "_SUM"

    ## make the folder for the full audio
    if os.path.isdir(PATH_DB + folder + '/' + song_name):
        print(f"Folder {song_name} already exists. deleting")
        os.removedirs(PATH_DB + folder + '/' + song_name)
    os.makedirs(PATH_DB + folder + '/' + song_name)

    ## copy the full audio to the folder
    os.system(f"cp {full_audio} {PATH_DB + folder + '/' + song_name}")

    ## rename the full audio to the song name, if exists, delete:
    if os.path.isfile(PATH_DB + folder + '/' + song_name + '/' + song_name + '.wav'):
        os.system(f"rm {PATH_DB + folder + '/' + song_name + '/' + song_name + '.wav'}")

    
    os.system(f"mv {PATH_DB + folder + '/' + song_name + '/' + folder + '_MIX.wav'} {PATH_DB + folder + '/' + song_name + '/' + song_name + '.wav'}")




def split_audios(stem: str, duration=10) -> list:
    '''
    Splits the audio into chunks of duration seconds.

    params:
    - stem: str, path to the stem
    - duration: int, duration of the audio chunks in seconds

    return:
    - chunks: list, list of audio chunks
    '''

    #Load in stems
    stem_audio, sr = librosa.load(stem, mono=False, sr=SR)

    # Transpose to have time as first dimension
    stem_audio = stem_audio.T

    # get total number of samples and number of chunks
    n_samples = len(stem_audio)
    # print(n_samples)
    n_chunks = n_samples // (sr * duration)
    # print(n_chunks)
    chunks = []

    # Split into chunks
    for i in range(n_chunks):
        start = i * sr * duration
        end = (i + 1) * sr * duration
        stem_chunk = stem_audio[start:end]
        chunks.append(stem_chunk)

    return chunks


def write_audio(chunks: list, stemname: str, destination: str) -> None:
    '''
    Writes the audio chunks to the destination.

    params:
    chunks: list, list of audio chunks
    stemname: str, name of the stem
    destination: str, destination to write the audio chunks

    return:
    None
    '''

    # Write chunks to destination
    chunks_written = 0
    for i, chunk in enumerate(chunks):        
        sf.write(f'{destination}/{stemname}_{i}.wav', chunk, SR)
        chunks_written += 1

    # print(f"Written {chunks_written} chunks for {stemname}.")

def song_has_stem(folder: str, stem_type: str, PATH_DB: str) -> bool:
    '''
    Checks if the song has the stem type in the folder.

    params:
    folder: str, name of the folder (song) in the db
    stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    PATH_DB: str, path to the db

    return:
    bool, True if the song has the stem type, False otherwise
    '''
    if not os.path.isdir(PATH_DB + folder):
        return False

    stem_folders = os.listdir(PATH_DB + folder)
    desired_stem_dir = [folder + '/' + stem for stem in stem_folders if stem==f"{stem_type}_INSTR" or stem==f"{stem_type}_INSTR_SUM"]

    if len(desired_stem_dir) == 0:
        return False
    else:
        return True


def split_per_folder(folder: str, PATH_DB: str, PATH_Train: str, stem_type: str, seconds: int =10) -> None:
    '''
    Splits the audio in the folder into chunks of duration seconds.

    params:
    folder: str, name of the folder (song) in the db
    PATH_DB: str, path to the db
    PATH_Train: str, path to the training data
    stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    seconds: int, number of seconds to split the audio into

    return:
    None
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

    # if song does not have stem type, skip
    if not song_has_stem(folder, stem_type, PATH_DB):
        print(f"Song {folder} does not have {stem_type} stem. Skipping.")
        return None

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
        perm_path_from_db = PATH_DB + stem 

        # sould only be one perm stem 
        perm = os.listdir(perm_path_from_db)
        assert len(perm) == 1, "More than one perm stem found."
        perm_path_from_db = perm_path_from_db + '/' + perm[0]

        
        # split perm stem into chunks
        perm_split = split_audios(perm_path_from_db, seconds)

        ## write perm stem
        write_audio(perm_split, f"{stem_type}_{i}", TRAIN_PATH)


def generate_data(PATH_DB: str, PATH_Train: str, stem_type: str, seconds: int =10) -> None:
    '''
    Generates the training data for a given stem type.

    params:
    PATH_DB: str, path to the db
    PATH_Train: str, path to the training data
    stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    seconds: int, number of seconds to split the audio into

    return:
    None
    '''
    folders = os.listdir(PATH_DB)
    for folder in folders:

        # add full audio to perms, might error if full audio already exists, in that case, just go to split_per_folder
        try:
            add_full_audio_to_perms(folder, PATH_DB)
        except Exception as e:
            print(f"Error adding full audio to perms for {folder}. {e}")

        # split the audio into chunks
        split_per_folder(folder, PATH_DB, PATH_Train, stem_type, seconds)


def clean_training_data(PATH_Train: str, stem_type: str, folder: str=None) -> None:
    '''
    Cleans the training data set for a given stem type, removing all files in the y_true and y_train folders.

    params:
    PATH_Train: str, path to the training data
    stem_type: str, type of stem to split (e.g. vocals, drums, bass, midrange)
    folder: str, name of the folder (song) in the db (OPTIONAL, if None, clean all folders in the training data)

    return:
    None
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