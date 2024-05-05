import os
import yaml

PATH = "Datasets/MedleyDB/Metadata/"

## if 'has_bleed' is yes, then the track has bleed. 

## yaml files of the form AmadeusRedux_SchubertMovement3_METADATA.yaml -> name of the track is AmadeusRedux_SchubertMovement3

def has_bleed():
    has_bleed = {}
    for file in os.listdir(PATH):
        if file.endswith("_METADATA.yaml"):
            with open(PATH + file, 'r') as stream:
                data_loaded = yaml.safe_load(stream)
                
                if data_loaded.get('has_bleed') == 'yes':
                    has_bleed[file.split('_METADATA')[0]] = True

    
    print(has_bleed)
    return has_bleed
                    
                    
has_bleed()
                    