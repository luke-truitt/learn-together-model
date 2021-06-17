import glob
import numpy as np


def load_files(classes, verbose=False):
    
    files = glob.glob('data/{}/train/*'.format(classes))
    
    return files

def build_cat_dog_sets(files, verbose=False):
    cat_files = [fn for fn in files if 'cat.' in fn] 
    dog_files = [fn for fn in files if 'dog.' in fn] 

    return {'cat': cat_files, 'dog': dog_files}

def produce_tvt(files, splits = [0.6, 0.2, 0.2], verbose=False):
    class_splits = {}
    MAX_FILES = 2500
    for k in files.keys():
        split_dict = {}
        total_len = MAX_FILES if len(files[k]) > MAX_FILES else len(files[k])
        train_size = int(total_len*splits[0])
        val_size = int(total_len*splits[1])
        test_size = int(total_len*splits[2])

        train = np.random.choice(files[k], size=train_size, replace=False)
        files[k] = list(set(files[k]) - set(train))
        val = np.random.choice(files[k], size=val_size, replace=False) 
        files[k] = list(set(files[k]) - set(val))
        test = np.random.choice(files[k], size=test_size, replace=False)  

        if verbose:
            print("Dataset Shapes:", train.shape, val.shape, test.shape)

        split_dict = {'train': train, 'val': val, 'test': test}

        class_splits[k] = split_dict
    
    return class_splits

def build_dataset(classes, verbose=False):
    files = load_files(classes, verbose=verbose)
    
    class_dict = {}
    if classes == "cat_dog":
        if verbose:
            print("Loading Cat Dog data")
        class_dict = build_cat_dog_sets(files, verbose=verbose)

    if len(class_dict) > 0:
        if verbose:
            print("loaded data successfully")
        splits = produce_tvt(class_dict, verbose=verbose)

    return splits