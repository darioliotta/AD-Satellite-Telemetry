###############################
# Imports # Imports # Imports #
###############################

import os
import pickle
import ast
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Union
from datetime import datetime
from TranADPlus.src import omnianomaly
import TranADPlus.src.global_config as global_config

# Load config
config = global_config.config

#########################################
# Generic Functions # Generic Functions #
#########################################

def generate_DSN_anomaly_contribution_text(anomaly_scores: np.ndarray, source_track: np.ndarray, label: list[int, int], label_num: int = None) -> str:
    """
    `anomaly_scores`: Per-parameter anomaly values (np.ndarray)
    `source_track`: Source track corresponding to anomaly values from `anomaly_scores` (np.ndarray)
    `label`: Indices of labeled anomalous regions (list[int, int])
    `label_num`: What anomaly are we printing data for (int)  

    Produce text like so:

    ```text
    Anomaly 8 parameters:
        PARAM1 (54.49%)
        PARAM2 (23.99%)
        PARAM3 (4.35%)
    Contribute to 82.83% of anomaly score at times
        "2023-05-29 19:23:01" to "2023-05-29 19:24:37"
        (index 3125 to 3144)
    ```

    Returns: `str`

    NOT UNIT TESTED  
    Reason: This is just printing data, not critical in functionality
    """
    out_str = ""

    sorted_list = get_dsn_single_anomaly_parameter_contribution_named_sorted(anomaly_scores, label)

    if label_num is not None:
        out_str += f"Anomaly {label_num} parameters:\n"
    else:
        out_str += f"Anomaly parameters:\n"
    date_start = datetime.utcfromtimestamp(source_track[label[0],0]).strftime('%Y-%m-%d %H:%M:%S')
    date_end = datetime.utcfromtimestamp(source_track[label[1],0]).strftime('%Y-%m-%d %H:%M:%S')

    cum_sum = 0
    for i in range(0,3):
        percent_contrib = sorted_list[i][1]
        cum_sum += percent_contrib
        out_str += f"  {sorted_list[i][0]} ({percent_contrib*100:0.2f}%)\n"
    out_str += f"Contribute to {cum_sum*100:0.2f}% of anomaly score at times\n"
    out_str += f"  \"{date_start}\" to \"{date_end}\"\n"
    out_str += f"  (index {label[0]} to {label[1]})\n"

    return out_str

def downsample(input_arr: np.ndarray, m: int) -> np.ndarray:
    '''
    Given an (n,d) numpy array, down-sample the array by calculating the median of every `m` data points  

    `input_arr`: (n,d) numpy array to downsample along row-dimension (np.ndarray)  
    `m`: every `m` data points we take the median of the `input_arr` (int)  

    Returns: `np.ndarray`

    Tested (`test_helpers.test_downsample`)
    '''    
    # 0: calculate d
    d = input_arr.shape[1]
    # 1: cut off end
    new_len = input_arr.shape[0] - (input_arr.shape[0] % m)
    A = input_arr[0:new_len,:]
    # 2: reshape
    B = A.T
    C = B.reshape(d, -1, m)
    # 3: median
    D = np.median(C ,2)
    # 4: Reshape again
    E = D.T
    return E

def merge_tracks(track_paths: list[str], window_sz: int = None, padding: bool = False) -> None:
    """
    Given a list of tracks (or anomalies) by filename, create a single merged
    track, accounting for padding

    `track_paths`: List of absolute file paths of tracks to merge together into one track (list(str))  
    `window_sz`: Size of window we expect to use in training (int)  
    `padding`: If `True`, pad the start of the track if `window_sz > 1` (bool)  

    Returns: `None`

    Tested (`test_helpers.test_merge_tracks`)
    """
    # Store separate tracks from original files
    track_l = list()

    # Combine all tracks into a single numpy array
    for idx in range(len(track_paths)):
        # Load data
        track = np.load(track_paths[idx])

        # Append list of all tracks
        if padding:
            # Add padding such that first window is just the first timestep repeated
            # next window will have the next timestep at the end
            track_padding = np.tile(track[0], (window_sz-1,1))
            track = np.vstack([track_padding, track])
            track_l.append(track)
        else:
            # No preprocessing
            track_l.append(track)

    # Save big track to class
    big_track = np.vstack(track_l, dtype=np.float32)

    return big_track

def parse_dim_feedforward(features: int, input: Union[str, int]) -> int:
    """
    Helper function to get dynamic feed forward values, `features` is given from the dataset number of parameters and `input` is either an integer (so we return it) or a float followed by `x`, like `2.1x`, so we return `2.1*features` as the feedforward dimension.

    `features`: Features/number of parameters of the dataset (int)  
    `input`: Either an integer or a float with an `x` at the end (`2.1x`) (int or str)  

    Returns: `int`

    Tested (`test_helpers.test_parse_dim_feedforward`)
    """

    try:
        # See if input is a value
        output = int(input)
        return output
    except:
        # Input is not a value, continue
        pass

    if 'x' == input[-1]:
        output = int(features*float(input.split('x')[0]))
        return output
    else:
        raise Exception(f"ERROR: Input {input} not valid, either need int or string of type \'Fx\' where F is a float")

def get_n_features_from_dataset(dataset: Literal['DSN_1k', 'DSN_100', 'MSL', 'DSN', 'SWaT', 'WADI', 'SMD']) -> int:
    """
    Return number of features in the *preprocessed* dataset.

    `dataset`: Input dataset to get features from (str)

    Returns: `int`

    Tested (`test_helpers.get_n_features_from_dataset`)
    """
    # Input check
    if dataset not in ['MSL', 'SMAP', 'DSN_100', 'DSN_1k', 'SWaT', 'WADI', 'SMD']:
        raise Exception(f"Dataset {dataset} is not valid, use MSL, SMAP, SWaT, WADI, SMD, DSN_100, or DSN_1k.")

    track_dir = os.path.join(config.preprocess_save_dir, dataset, 'Both', 'Tracks')
    filenames = os.listdir(track_dir)
    filenames.sort()
    track_path = os.path.join(track_dir, filenames[0])

    with open(track_path, 'rb') as file:
        track = np.load(file)
    
    return track.shape[1]

def get_n_one_hot_features_from_dataset(dataset: Literal['DSN_1k', 'DSN_100', 'MSL', 'DSN', 'SWaT', 'WADI', 'SMD']) -> int:
    """
    Return number of features after one-hot encoding
    for the given dataset. Dataset needs to be
    parsed first since we load the `byte_vals.pkl`
    file

    `dataset`: Input dataset to get features from (str)

    Returns: `int`

    Tested (`test_helpers.get_n_one_hot_features_from_dataset`)
    """
    # Input check
    if dataset not in ['MSL', 'SMAP', 'DSN_100', 'DSN_1k', 'SWaT', 'WADI', 'SMD']:
        raise Exception(f"Dataset {dataset} is not valid, use MSL, SMAP, SWaT, WADI, SMD, DSN_100, or DSN_1k.")

    byte_val_path = os.path.join(config.parse_save_dir, dataset, 'Both', 'pkl', f'byte_vals.pkl')
    with open(byte_val_path, 'rb') as file:
        byte_vals = pickle.load(file)
    
    return get_n_one_hot_features_from_byte_dict(byte_vals)

def get_n_one_hot_features_from_byte_dict(byte_vals: list[list[any]]) -> int:
    '''
    Given a byte_vals list of lists, return the number of features
    if we were to one-hot encode the data

    `byte_vals`: Input dataset to get features from (str)  

    Returns: `int`

    Tested (`test_helpers.get_n_one_hot_features_from_byte_dict`)
    '''
    byte_sum = 0
    non_byte_sum = 0
    for i in range(len(byte_vals)):
        n_byte_vals = len(byte_vals[i])
        if n_byte_vals == 0:
            non_byte_sum += 1
        else:
            byte_sum += n_byte_vals
    return byte_sum + non_byte_sum


def replace_byte_constant(df: pd.DataFrame, byte_value: float = 0.0) -> pd.DataFrame:
    '''
    Replace all byte values by a constant float

    `df`: Data frame that will have bytes replaced with the constant `byte_value` (pd.DataFrame)  
    `byte_value`: The constant value that replaces bytes in a pandas data frame (float)  

    Returns: `pd.DataFrame`

    Tested (`test_helpers.replace_byte_constant`)
    '''
    df = df.copy()
    for column, type in enumerate(df.dtypes):
        if type == np.dtype('O'):
            df.iloc[:,column] = byte_value
    return df

def replace_byte_dict(df: pd.DataFrame, byte_vals: list[list[any]]) -> pd.DataFrame:
    '''
    Replace all byte values their index in the parameter's byte_vals

    `df`: Data frame that will have bytes replaced with the constant `byte_value` (pd.DataFrame)  
    `byte_vals`: List of all encountered byte values per-parameter (list)  

    ex. byte_vals: [[], [b'X',b'Y']]
    The parameter 0 is not a byte value parameter
    The parameter 1 is a byte value parameter, all values b'X' get replaced with 0
      and all values b'Y' get replaced with 1

    Returns: `pd.DataFrame`

    Tested (`test_helpers.replace_byte_dict`)
    '''
    df = df.copy()
    for column, type in enumerate(df.dtypes):
        if type == np.dtype('O'):
            map_fn = lambda byte_val: byte_vals[column].index(byte_val)
            df.iloc[:,column] = df.iloc[:,column].map(map_fn)
    return df

def replace_byte_one_hot(df: pd.DataFrame, byte_vals: list[list[any]]) -> pd.DataFrame:
    '''
    Replace all byte values by their one-hot encoding, expanding the number of features

    `df`: Data frame that will have bytes replaced with the constant `byte_value` (pd.DataFrame)  
    `byte_vals`: List of all encountered byte values per-parameter (list)  

    ex. byte_vals: [[], [b'X',b'Y']]
    The parameter 0 is not a byte value parameter
    The parameter 1 is a byte value parameter, its column gets turned into two
      where the values are one-hot encoded

    Returns: `pd.DataFrame`

    Tested (`test_helpers.replace_byte_one_hot`)
    '''
    # Copy track
    df = df.copy()
    # Update column names
    df.columns = pd.RangeIndex(0, df.shape[1])
    # Get data types of each column
    types = df.dtypes
    # Store new dataframes and concat at the end
    new_frames = list()
    # Iterate last column to recent one since we're dropping columns
    for column in reversed(range(len(types))):
        # Only replace byte objects
        if types[column] == np.dtype('O'):
            # Get possible byte values for this column and how many
            c_bvs = byte_vals[column]
            n_bvs = len(c_bvs)
            # Convert each byte value in this column to its integer value
            map_fn = lambda byte_val: c_bvs.index(byte_val)
            col_l = df.iloc[:,column].map(map_fn).to_list()
            # Add all possible byte values to head of list
            col_l = list(range(n_bvs)) + col_l
            # Get one-hot representation of the column
            one_hot = pd.get_dummies(col_l,dtype=int)
            # Remove prepended rows
            one_hot = one_hot.iloc[n_bvs:,:]
            # Drop the original byte-to-int column
            df = df.drop(column, axis='columns')
            # Update columns names
            col_names = [f"{column}_{i}" for i in range(n_bvs)]
            one_hot.columns = col_names
            # Update index names
            one_hot.reset_index(inplace=True, drop=True)
            # Store new dataframe to list to append at the end
            new_frames.insert(0, one_hot)
    # Save remaining non-byte dataframe and add to list
    new_frames.insert(0, df)
    df = pd.concat(new_frames, axis=1)
    return df

def gen_byte_dict(byte_vals: dict[bytes]) -> dict:
    """
    Generate dictionary for byte values
    Each unique byte value gets the next integer, starting from 0

    `byte_vals`: the individual bytes that need to be assigned an integer

    Returns: `dict()`

    Tested (`test_helpers.gen_byte_dict`)
    """
    counter = 0
    byte_dict = dict()
    for param_num in range(len(byte_vals)):
        for val_num in range(len(byte_vals[param_num])):
            # Get byte value
            byte = byte_vals[param_num][val_num]
            # See if value exists if not, add it
            tmp = byte_dict.get(byte, None)
            if tmp is None:
                # Byte not in dictionary
                byte_dict[byte] = counter
                counter += 1
    return byte_dict

def normalize3(a: np.ndarray, min_a: int = None, max_a: int = None) -> tuple[np.ndarray, float, float]:
    """
    From TranAD/preprocess.py: This is what preprocesses SMAP data, scaled each parameter to be in [0,1]

    Returns: (np.ndarray, float, float)

    Tested (`test_helpers.normalize3`)
    """
    if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
    diff = max_a - min_a # different from TranAD, variable epsilon
    epsilon = 0.0001
    diff[diff == 0] = epsilon
    return (a - min_a) / (diff), min_a, max_a

def get_dataset_dfs(dataset: Literal['DSN_100', 'DSN_1k', 'SMAP', 'WADI', 'SWaT', 'MSL'], split: Literal['Train', 'Test', 'Both'] = 'Both', track_names: list[str] = [])\
    -> tuple[list[pd.DataFrame], list[list[int, int]], list[str]]:
    """
    Convert track names of a dataset to:
        a list of dataframes and
        a list of labels (ie, a list of start/end indices for each anomaly)
        a list of names for each dataframe/label

    `dataset`: String of dataset (str)  
    `split`: Split of dataset (str)  
    `track_names`: Name of tracks that we're getting data frames for  

    Returns: tuple[list[pd.DataFrame], list[list[int, int]], list[str]]

    Tested (`test_helpers.test_get_dataset_dfs`)
    """
    # Input check
    if dataset not in ['MSL', 'SMAP', 'DSN_100', 'DSN_1k', 'SWaT', 'WADI']:
        raise Exception(f"Dataset {dataset} is not valid, use MSL, SMAP, or DSN.")

    # Output lists
    tracks = list()
    labels = list()

    # Iterate over track names to get labels
    for i in range(len(track_names)):
        track_name = track_names[i]
        # Read label
        label_path = os.path.join(config.val_dict[f'{dataset.lower()}_data_dir'], split, 'Labels', f'{track_name}.l.pkl')
        with open(label_path, 'rb') as file:
            label_l = pickle.load(file)
        labels.append(label_l)
        # Read track
        track_path = os.path.join(config.val_dict[f'{dataset.lower()}_data_dir'], split, 'Tracks', f'{track_name}.df.pkl')
        with open(track_path, 'rb') as file:
            track_df = pickle.load(file)
        tracks.append(track_df)
    
    return tracks, labels

def get_all_dataset_dfs(dataset: Literal['DSN_100', 'DSN_1k', 'MSL', 'SMAP'], split: Literal['Train', 'Test', 'Both'] = 'Both')\
    -> tuple[list[pd.DataFrame], list[list[int, int]], list[str]]:
    """
    Convert dataset of tracks to:
        a list of dataframes and
        a list of labels (ie, a list of start/end indices for each anomaly)
        a list of names for each dataframe/label

    `dataset`: String of dataset (str)
    `split`: Split of dataset (str)

    Returns: `tuple[list[pd.DataFrame], list[list[int, int]], list[str]]`

    Tested (`test_helpers.get_all_dataset_dfs`)
    """
    # Input check
    if dataset not in ['MSL', 'SMAP', 'DSN_100', 'DSN_1k']:
        raise Exception(f"Dataset {dataset} is not valid, use MSL, SMAP, DSN_1k, or DSN_100.")

    # Output lists
    tracks = list()
    labels = list()
    track_names = get_dataset_split_track_names(dataset, 'Both')

    tracks, labels = get_dataset_dfs(dataset, split, track_names)
    
    return tracks, labels, track_names

def load_l_from_path(path: str) -> np.ndarray:
    """
    Given a path to a <file>.l.pkl, load the list

    `path`: Path of the pickled list file (str)

    Returns: `np.ndarray`
    
    Tested (`test_helpers.load_l_from_path`)
    """
    with open(path, 'rb') as file:
        l = pickle.load(file)
    return l

def load_npy_from_path(path: str) -> np.ndarray:
    """
    Given a path to a <file>.npy.pkl, load the numpy array

    `path`: Path of the pickled numpy file (str)

    Returns: `np.ndarray`
    
    Tested (`test_helpers.load_npy_from_path`)
    """
    with open(path, 'rb') as file:
        npy = pickle.load(file).to_numpy()
    return npy

def load_df_from_path(path: str) -> pd.DataFrame:
    """
    Given a path to a <file>.df.pkl, load the pandas
    
    `path`: Path of the pickled data frame file (str)

    Returns: `pd.DataFrame`
    
    Tested (`test_helpers.load_df_from_path`)
    """
    with open(path, 'rb') as file:
        df = pickle.load(file)
    return df

def get_dataset_split_absolute_file_paths(dataset: Literal['SMAP', 'MSL', 'DSN_100', 'DSN_1k', 'SWaT', 'WADI', 'SMD'],\
                                          split: Literal['Train', 'Test', 'Both']) -> list[str]:
    """
    Returns the paths of the files for the specified dataset

    `dataset`: String of dataset (str)  
    `split`: Split of dataset (str)  

    Returns: `list[str]`
    
    Tested (`test_helpers.get_dataset_split_absolute_file_paths`)
    """
    # Input check
    if dataset not in ['MSL', 'SMAP', 'DSN_100', 'DSN_1k', 'SWaT', 'WADI', 'SMD']:
        raise Exception(f"Dataset {dataset} is not valid, use MSL, SMAP, DSN_1k, DSN_100, SWaT, WADI, or SMD.")

    if dataset in ['SMAP', 'MSL', 'SWaT', 'SMD', 'WADI', 'DSN_100', 'DSN_1k']:
        if split not in ['Train', 'Test', 'Both']:
            raise Exception(f"Split {split} not available for {dataset}, only \"Train\", \"Test\", or \"Both\"")
    else:
        raise Exception(f"Dataset {dataset} invalid.")

    fns = get_dataset_split_track_names_full(dataset, split)
    paths = [os.path.join(config.val_dict[f'{dataset.lower()}_data_dir'], split, 'Tracks', fn) for fn in fns]
    return paths

def get_dataset_split_track_names_full(dataset: Literal['SMAP', 'MSL', 'DSN_100', 'DSN_1k', 'SWaT', 'WADI', 'SMD'],\
                                       split: Literal['Both', 'Train', 'Test']) -> list[str]:
    """
    Returns the file names of the files for the specified dataset and split with extensions
    
    `dataset`: String of dataset (str)  
    `split`: Split of dataset (str)  

    Returns: `list[str]`
    
    Tested (`test_helpers.get_dataset_split_track_names_full`)
    """
    # Input check
    if dataset not in ['MSL', 'SMAP', 'SWaT', 'WADI', 'SMD', 'DSN_100', 'DSN_1k']:
        raise Exception(f"Dataset {dataset} is not valid, use MSL, SMAP, SWaT, WADI SMD, DSN_1k, or DSN_100.")

    dataset_path = os.path.join(config.val_dict[f'{dataset.lower()}_data_dir'], split, 'Tracks')
    filenames = os.listdir(dataset_path)
    filenames.sort()
    return filenames

def get_dataset_split_track_names(dataset: Literal['SMAP', 'MSL', 'DSN_100', 'DSN_1k', 'SWaT', 'WADI', 'SMD'],\
                                  split: Literal['Both', 'Train', 'Test']) -> list[str]:
    """
    Returns the file names of the files for the specified dataset and split without extensions
    
    `dataset`: String of dataset (str)  
    `split`: Split of dataset (str)  

    Returns: `list[str]`
    
    Tested (`test_helpers.get_dataset_split_track_names`)
    """
    # Input check
    if dataset not in ['MSL', 'SMAP', 'DSN_100', 'DSN_1k', 'SWaT', 'WADI', 'SMD']:
        raise Exception(f"Dataset {dataset} is not valid, use MSL, SMAP, SWaT, WADI, SMD, DSN_1k, or DSN_100.")

    return [filename.split('.')[0] for filename in get_dataset_split_track_names_full(dataset, split)]

def convert_str_to_int_list(str_input: str) -> list[int]:
    """
    Converts a str like \"[[1, 2], [2, 3]]\" to a list of integers [[1, 2], [2, 3]].

    `str_input`: The input string (str)  

    Returns: `list[int]`
    
    Tested (`test_helpers.convert_str_to_int_list`)
    """
    return ast.literal_eval(str_input)

def flatten_list(list_in: list) -> list:
    """
    Convert list like [[1,2],[2,3]] to [1,2,2,3].

    `list_in`: input list to flatten (list)  
    
    Tested (`test_helpers.flatten_list`)
    """
    return [item for sublist in list_in for item in sublist]

def make_dir_tree(dir_in: list) -> None:
    """
    Creates directories and parent directories defined in `dir_in` which is
    a list of paths, like ['../new_dir/new_subdir', '../new_dir2/new_subdir2']

    If `dir_in` is a string, just create the directory

    `dir_in`: string or list of strings containing directories to create (list(str) | str)  

    Returns: `None`

    Tested (`test_helpers.make_dir_tree`)
    """

    if type(dir_in) is list:
        for cur_dir_in in dir_in:
            Path.mkdir(Path(cur_dir_in), exist_ok = True, parents = True)
    elif type(dir_in) is str:
        Path.mkdir(Path(dir_in), exist_ok = True, parents = True)
    else:
        raise Exception(f"Unknown type encountered: {type(dir_in)}")

    return None

def shift_anoms(anoms: Union[list[list[int, int]], np.ndarray], shift_amount: int) -> list[list[int, int]]:
    """
    Add `shift_amount` to every element in provided anoms

    `anoms`: list of anomalies with start and end indices as integers: `[[start1, end1], [start2, end2]]` (list)  
    `shift_amount`: Amount to add to each index (int)  

    Returns: `list[list[int, int]]`
    
    Tested (`test_helpers.test_shift_anoms`)
    """
    anoms = np.asarray(anoms)
    new_anoms = list()
    for i in range(anoms.shape[0]):
        # Get one anomaly at a time
        anom = anoms[i]
        new_anoms.append([anom[0] + shift_amount, anom[1] + shift_amount])
    return new_anoms

def extract_anoms(anoms: Union[list[list[int, int]], np.ndarray], window_start_idx: int, window_end_idx: int) -> list[list[int,int]]:
    """
    Given a window start index and end index (inclusive) and a numpy array of anomaly values,
    return anomalies in that window with appropriate indices.

    `anoms`: list of anomalies with start and end indices as integers: `[[start1, end1], [start2, end2]]` (list)  
    `window_start_idx`: starting index for window we want to extract anomalies from (int)  
    `window_end_idx`: ending index for window we want to extract anomalies from (int)  
    
    Ex. window start: 10, window end: 15, anomalies: [[0,13],[13,20]]
    Returns: `list[list[int,int]] (ex. [[0,3],[3,5]])`

    Tested (`test_helpers.text_extract_anoms`)
    """
    anoms = np.asarray(anoms)
    window_size = window_end_idx - window_start_idx + 1
    # Update anomaly values
    new_anoms = list()
    for i in range(anoms.shape[0]):
        # Get one anomaly at a time
        anom_range = anoms[i]

        # Skip anomaly if it's not in the window
        if anom_range[1] < window_start_idx or anom_range[0] > window_end_idx:
            continue

        # Anomaly is in window, make adjustments
        new_min = max(0, anom_range[0] - window_start_idx)
        new_max = min(window_size-1, anom_range[1] - window_start_idx)
        new_anoms.append([new_min, new_max])
    return new_anoms

###################################
# Evaluate model # Evaluate model #
###################################

def gen_TranAD_predictions(model, loss_fn, dataloader, alpha=0.5, beta=0.5, device='cpu') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Given a trained TranAD model, a loss function, and a dataloader,
    generate predictions for it.

    `model`: TranAD model  
    `loss_fn`: TranAD loss function  
    `dataloader`: Windowed data loader for TranAD  
    `alpha`: Alpha for scaling auto-encoder outputs (float)  
    `beta`: Beta for scaling auto-encoder outputs (float)  
    `device`: Which device PyTorch should use (str)  

    Returns: `tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`

    NOT UNIT TESTED  
    Reason: This function is complicated and isn't trivial to evaluate functionality
    '''
    # Output variables
    loss, pred_y, elem, true_y, anoms = (list(), list(), list(), list(), list())

    # Generate predictions
    torch.zero_grad = True
    with torch.no_grad():
        model.eval()
        # Iterate over dataset
        for windows, anoms_tmp in dataloader:
            # Convert anoms to numpy
            anoms_tmp = anoms_tmp.numpy()
            # Permute so that its shape is (ws x bs x #params)
            windows = windows.to(device).permute(1,0,2)
            # Extract last timestep - the one we're predicting
            elem_tmp = windows[-1,:,:].unsqueeze(0)
            # Pass through model
            z = model(windows, elem_tmp) # Z is tuple of x1 (O1) and x2 (O2)
            z = alpha*z[0] + beta*z[1]
            # Calculate loss
            loss_tmp = loss_fn(z, elem_tmp)[0]
            pred_y_tmp = z
            elem_tmp = elem_tmp
            true_y_tmp = elem_tmp[0]

            # Save values for whole dataset
            loss.append(loss_tmp.detach().cpu().numpy())
            pred_y.append(pred_y_tmp[0].detach().cpu().numpy())
            elem.append(elem_tmp[0].detach().cpu().numpy())
            true_y.append(true_y_tmp.detach().cpu().numpy())
            # Take only latest time slice of anomaly
            if dataloader.dataset.padding:
                anoms.append(anoms_tmp[:,-1,:])
            else:
                anoms.append(anoms_tmp[:,0,:])

    loss = np.vstack(loss)
    pred_y = np.vstack(pred_y)
    elem = np.vstack(elem)
    elem = np.expand_dims(elem, 0)
    true_y = np.vstack(true_y)
    anoms = np.vstack(anoms)

    return loss, pred_y, elem, true_y, anoms

def evaluate_TranAD(model, train_dataloader, test_dataloader, alpha=0.5, beta=0.5, pot_q=1e-5, device='cpu') -> dict:
    """
    Evaluate TranAD model

    `model`: TranAD model  
    `loss_fn`: TranAD loss function  
    `train_dataloader`: Windowed training data loader for TranAD  
    `test_dataloader`: Windowed testing data loader for TranAD  
    `alpha`: Alpha for scaling auto-encoder outputs (float)  
    `beta`: Beta for scaling auto-encoder outputs (float)  
    `pot_q`: POT q parameter (float)  
    `device`: Which device PyTorch should use (str)  

    Returns: `dict`

    NOT UNIT TESTED  
    Reason: This function is complicated and isn't trivial to evaluate functionality
    """
    # Should this be somewhere else?
    loss_fn = nn.MSELoss(reduction='none')

    # Infer testing data
    loss_test, pred_y_test, elem_test, true_y_test, anoms_test = \
        gen_TranAD_predictions(model, loss_fn, test_dataloader, alpha=alpha, beta=beta, device=device)
    loss_train, pred_y_train, elem_train, true_y_train, anoms_train = \
        gen_TranAD_predictions(model, loss_fn, train_dataloader, alpha=alpha, beta=beta, device=device)

    # Create POT and evaluate model
    loss_train_final, loss_test_final = np.mean(loss_train, axis=1), np.mean(loss_test, axis=1)
    labels_test_final = (np.sum(anoms_test, axis=1) >= 1) + 0
    result, _, _ = omnianomaly.pot_eval(loss_train_final, loss_test_final, labels_test_final, pot_q)

    # Brute force search
    start = np.min(loss_test_final)
    end = np.mean(loss_test_final)+np.std(loss_test_final)*3
    print(f"bf_search: ({start},{end})")
    bf_eval = omnianomaly.bf_search(loss_test_final, labels_test_final, start=start, end=end, step_num=200, verbose=False)
    result.update(bf_eval)

    return result

def evaluate_USAD(model, test_loader, train_loader, alpha=.5, beta=.5, pot_q=1e-5, device='cpu', do_print=True) -> dict:
    """
    Evaluate USAD model

    `model`: USAD model  
    `loss_fn`: USAD loss function  
    `train_dataloader`: Windowed training data loader for USAD  
    `test_dataloader`: Windowed testing data loader for USAD  
    `alpha`: Alpha for scaling auto-encoder outputs (float)  
    `beta`: Beta for scaling auto-encoder outputs (float)  
    `pot_q`: POT q parameter (float)  
    `device`: Which device PyTorch should use (str)

    Returns: `dict`

    NOT UNIT TESTED  
    Reason: This function is complicated and isn't trivial to evaluate functionality
    """
    # Step 1: obtain results
    loss_test=[]
    anoms_test=[]
    loss_train=[]
    anoms_train=[]
    with torch.no_grad():
        # Test data
        for windows, anoms in test_loader:
            windows = windows.to(device).reshape(windows.shape[0], -1)
            w1=model.decoder1(model.encoder(windows))
            w2=model.decoder2(model.encoder(w1))
            loss_test_tmp = alpha*torch.mean((windows-w1)**2,axis=1)+beta*torch.mean((windows-w2)**2,axis=1)
            loss_test.append(loss_test_tmp.detach().cpu().numpy())
            # Take current time slice of anomaly, accounting for padding method
            if test_loader.dataset.padding:
                anoms = anoms[:,-1,:] 
            else:
                anoms = anoms[:,0,:]
            anoms_test.append(anoms.numpy())
        # Train data
        for windows, anoms in train_loader:
            windows = windows.to(device).reshape(windows.shape[0], -1)
            w1=model.decoder1(model.encoder(windows))
            w2=model.decoder2(model.encoder(w1))
            loss_train_tmp = alpha*torch.mean((windows-w1)**2,axis=1)+beta*torch.mean((windows-w2)**2,axis=1)
            loss_train.append(loss_train_tmp.detach().cpu().numpy())
            # Take current time slice of anomaly, accounting for padding method
            if train_loader.dataset.padding:
                anoms = anoms[:,-1,:] 
            else:
                anoms = anoms[:,0,:]
            anoms_train.append(anoms.numpy())
    # Step 2: concatenate everything
    loss_test=np.hstack(loss_test)
    anoms_test=np.vstack(anoms_test)
    labels_test = (np.sum(anoms_test, axis=1) >= 1) + 0
    loss_train=np.hstack(loss_train)
    anoms_train=np.vstack(anoms_train)
    labels_train = (np.sum(anoms_train, axis=1) >= 1) + 0

    # Step 3: calculate precision, recall, and F1 using POT and BF
    # Create POT and evaluate model
    result, _, _ = omnianomaly.pot_eval(loss_train, loss_test, labels_test, pot_q)

    start = np.min(loss_test)
    end = np.mean(loss_test)+np.std(loss_test)*3
    bf_eval = omnianomaly.bf_search(loss_test, labels_test, start=start, end=end, step_num=200, verbose=False)
    result.update(bf_eval)

    return result

###################################
# Training model # Training model #
###################################

def train_TranAD(model, epochs, dataloader, optimizer, scheduler, device, save_location = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Train TranAD model

    `model`: TranAD model  
    `epochs`: Number of epochs to train (int)
    `dataloader`: Windowed data loader for TranAD  
    `optimizer`: TranAD optimizer function  
    `scheduler`: TranAD scheduler function  
    `device`: PyTorch device to use (str)  
    `save_location`: Where to store model checkpoints (str)   

    Returns: `tuple[np.ndarray, np.ndarray]` 

    NOT UNIT TESTED  
    Reason: This function is complicated and isn't trivial to evaluate functionality
    """
    print("Training model")
    avg_losses = list()
    for epoch in range(epochs):
        epoch_losses = list()
        loss_fn = nn.MSELoss(reduction='none')
        for windows in dataloader:
            # Get sample
            windows = windows.to(device)
            # Permute so that its shape is (ws x bs x #params)
            windows = windows.permute(1,0,2)
            # Extract last timestep - the one we're predicting
            elem = windows[-1,:,:].unsqueeze(0)
            # Pass through model
            z = model(windows, elem) # Z is tuple of x1 (O1) and x2 (O2)
            # Calculate loss
            epoch_loss = (1 / (epoch+1)) * loss_fn(z[0], elem) + (1 - 1/(epoch+1)) * loss_fn(z[1], elem)
            epoch_loss = torch.mean(epoch_loss)
            epoch_losses.append(epoch_loss.item())
            optimizer.zero_grad()
            epoch_loss.backward(retain_graph=True)
            optimizer.step()
        epoch_avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch} avg. loss: {epoch_avg_loss:.4f}")
        if np.isnan(epoch_avg_loss):
            raise Exception("helpers: train_TranAD: epoch_avg_loss is nan")
        avg_losses.append(epoch_avg_loss)
        scheduler.step()
        # Save model
        if save_location is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()},
                save_location)
    return (list() if len(avg_losses) == 0 else avg_losses[-1], optimizer.param_groups[0]['lr'])

def train_USAD(epochs, model, train_loader, optimizer1, optimizer2, device) -> list[tuple[np.ndarray, np.ndarray]]:
    '''
    Train USAD model

    `model`: USAD model  
    `epochs`: Number of epochs to train (int)
    `dataloader`: Windowed data loader for USAD  
    `optimizer1`: USAD optimizer function (see paper)  
    `optimizer2`: USAD optimizer function (see paper)  
    `device`: PyTorch device to use (str)  
    `save_location`: Where to store model checkpoints (str)    

    Returns: `list[tuple[np.ndarray, np.ndarray]]`

    NOT UNIT TESTED  
    Reason: This function is complicated and isn't trivial to evaluate functionality

    Note: original code uses a split of the training data to calculate loss on validation
    data, I got rid of that here for now
    '''
    history = []
    for epoch in range(epochs):
        for windows in train_loader:
            # Get sample
            windows = windows.to(device)
            # Reshape so that the window is a vector
            windows = windows.reshape(windows.shape[0], -1)

            #Train AE1
            loss1,loss2 = model.training_step(windows,epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            #Train AE2
            loss1,loss2 = model.training_step(windows,epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
        result = {'train_loss1': loss1, 'train_loss2': loss2}
        model.epoch_end(epoch, result)
        history.append(result)
    return history

#############################################################################################
# DSN Helper Functions # DSN Helper Functions # DSN Helper Functions # DSN Helper Functions #
#############################################################################################

def get_dr_columns() -> list[str]:
    """
    Get columns names for discrepancy reports from provided raw drs
    
    NOT UNIT TESTED
    Reason: Simple list, hard-coded
    """
    l = ['INCIDENT_START_TIME_DT', 'INCIDENT_START_DATE_DT',
            'INCIDENT_START_TIME_NUM', 'INCIDENT_END_TIME_DT']
    return l

def get_dsn_single_anomaly_parameter_contribution(anomaly_scores: np.ndarray, label: list[int, int]) -> list[float]:
    """
    Given anomaly scores and a single anomaly range `[start1, end1]`, return list of how much each anomaly contributed in the same order as the parameter

    `anomaly_scores`: Per-parameter anomaly values (np.ndarray)
    `labels`: Indices of labeled anomalous regions (list[int, int])

    Returns: `list[float]`

    NOT UNIT TESTED  
    Reason: This is just a helper for printing data, not critical in functionality
    """
    anomaly_scores_label = anomaly_scores[label[0]:label[1]+1,:]
    anomaly_scores_sum = np.sum(anomaly_scores_label, axis=0)
    total_sum = np.sum(anomaly_scores_sum)
    ret_list = list(anomaly_scores_sum/total_sum)

    return ret_list

def split_dsn_tracks(ds: Literal['DSN_100', 'DSN_1k']) -> None:
    """
    Split DSN data into testing and training splits. Training set has 80% of total data but 0% of dr, testing data has 20% all data, with both non-dr and dr data.

    `ds`: Either `DNS_1k` or `DSN_100` (str)  

    Returns: `None`

    NOT UNIT TESTED  
    Reason: This function just makes two lists of tracks under certain conditions (see description), there are some asserts to make sure we're not losing data. Doesn't really need to be unit tested further since nothing is returned.  
    """

    # Load data
    mons, drs = load_raw_dsn_data(ds)

    # Determine which tracks have discrepancy reports
    np.random.seed(0)
    my_dict = gen_drs_dict(mons, drs)
    tracks_with_drs_indices = list(my_dict.keys())
    tracks = [mons[i][0] for i in range(len(mons))]
    tracks_w_drs = [mons[i][0] for i in tracks_with_drs_indices]
    if ds == 'DSN_1k':
        tracks.remove(17894088)
        tracks_w_drs.remove(17894088)
    tracks_wo_drs = list(set(tracks) ^ set(tracks_w_drs))

    # Split data: Training set has 80% of total data but 0% of dr, testing data has 20% of non-dr and dr data
    split_percent = 0.8

    n_train_tracks_w_drs = 0
    n_train_tracks_wo_drs = int(split_percent*len(tracks))
    n_test_tracks_w_drs = len(tracks_w_drs) - n_train_tracks_w_drs
    n_test_tracks_wo_drs = len(tracks_wo_drs) - n_train_tracks_wo_drs

    train_split_w_drs = np.random.choice(tracks_w_drs, n_train_tracks_w_drs, replace=False)
    train_split_wo_drs = np.random.choice(tracks_wo_drs, n_train_tracks_wo_drs, replace=False)
    train_split = list(train_split_w_drs) + list(train_split_wo_drs)

    test_split_w_drs = list(set(tracks_w_drs) ^ set(train_split_w_drs))
    test_split_wo_drs = list(set(tracks_wo_drs) ^ set(train_split_wo_drs))
    test_split = list(test_split_w_drs) + list(test_split_wo_drs)

    print(f"tracks ({len(tracks)}):")
    print(tracks)
    print(f"tracks_w_drs ({len(tracks_w_drs)}):")
    print(tracks_w_drs)
    print(f"tracks_wo_drs ({len(tracks_wo_drs)}):")
    print(tracks_wo_drs)
    print()

    print(f"Split percent: {split_percent*100}%")
    print(f"Total tracks: ({len(tracks_wo_drs)}+{len(tracks_w_drs)}={len(tracks)}):")
    print(tracks)
    print(f"Train tracks ({len(train_split_wo_drs)}+{len(train_split_w_drs)}={len(train_split)}):")
    print(train_split)
    print(f"Test tracks: ({len(test_split_wo_drs)}+{len(test_split_w_drs)}={len(test_split)}):")
    print(test_split)
    print()

    print(f"Train track %: {len(train_split)/len(tracks)*100:0.2f}%")
    print(f"Train dr %: {len(train_split_w_drs)/len(tracks_w_drs)*100:0.2f}%")
    print(f"Train non-dr %: {len(train_split_wo_drs)/len(tracks_wo_drs)*100:0.2f}%")
    print(f"Test track %: {len(test_split)/len(tracks)*100:0.2f}%")
    print(f"Test dr %: {len(test_split_w_drs)/len(tracks_w_drs)*100:0.2f}%")
    print(f"Test non-dr %: {len(test_split_wo_drs)/len(tracks_wo_drs)*100:0.2f}%")

    if not (len(train_split)+len(test_split) == len(tracks)):
        raise Exception(f"Train split ({len(train_split)}) + test split ({len(test_split)}) = ({len(train_split) + len(test_split)}) != tracks ({len(tracks)})")
    if len(set(train_split).intersection(test_split)) != 0:
        raise Exception(f"Train split contains tracks from test split")

    return None

def get_dsn_split_names(ds: Literal['DSN_1k', 'DSN_100']) -> list[list[str], list[str]]:
    '''
    Generated from parse_data.ipynb  
    Contains track names of each test and train split for DSN datasets  
    Train data has 80% of labeled (with dr) and 80% of unlabeled tracks (without dr)
    Test data has remaining 20% of labeled (with dr) and unlabeled tracks (without dr)

    `ds`: Either `DNS_1k` or `DSN_100` (str)  

    Returns: `list[list[str], list[str]]`

    NOT UNIT TESTED  
    Reason: These are hard-coded values
    '''
    if ds == 'DSN_1k':
        train_tracks = [17966720, 17505057, 17289440, 16961685, 17160831, 17169537, 18024689, 17452164, 16609954, 17356477, 17121010, 18282538, 17303964, 17207094, 16850478, 17339112, 17347848, 17190221, 17842222, 17393677, 16850509, 17265375, 17065687, 17737178, 16767465, 18279112, 17403311, 17823940, 17207098, 17274672, 17160836, 17965089, 17127211, 16609958, 17190209, 17610719, 17374234, 17168236, 16820008, 17355914, 17452152, 17495601, 17355915, 17231031, 17265373, 17894095, 17965085, 17317062, 18166690, 17169550, 17525534, 17006956, 17581014, 17389379, 16750851, 16854782, 16885601, 17253998, 17894100, 17303914, 16998324, 17610924, 17308282, 17352151, 17700928, 17571820, 17737177, 16998350, 17447340, 18165336, 17610863, 16982925, 17477747, 17816320, 18279111, 17347852, 17265374, 17274678, 17265370, 16850511, 17207079, 17645722, 17339659, 17081665, 17160846, 17094053, 17207086, 17894096, 17581011, 17964955, 17265372, 17816341, 17389370, 17376267, 17894097, 17482401, 17253992, 16697959, 17308285, 16891371, 17490272, 16609969, 17268359, 17472236, 17517263, 17448975, 17628164, 17207117, 17271992, 17385966, 17356468, 17525543, 16800086, 18007881, 16767467, 18165374, 17610746, 16880908, 17271993, 16944137, 17580997, 17628163, 16731154, 17352735, 17897345, 18007924, 17169472, 17432296, 17389353, 16998338, 17628167, 17517882, 17137571, 17160833, 17505065, 16761540, 18166669, 17477757, 17737176, 17065684, 17292759, 16961101, 17735157, 17003698, 16919928, 17517894, 17571819, 17477750, 17735161, 17371700, 16885608, 17581099, 16770596, 17894092, 16594758, 17201905, 17268385, 17389357, 16609965, 17735156, 17003724, 17393685, 17127175, 17352730, 17006828, 17137553, 17339651, 17743515, 17495600, 17729068, 17349873, 17265367, 17217082, 16944135, 17608808, 17885230, 17510165, 17885229, 17581007, 17137518, 17355909, 17857191, 17462880, 16993076, 17462277, 17268961, 17509623, 16801264, 17505058, 16854773, 17965094, 16919970, 17525524, 18281396, 17444121, 16982927, 17065674, 17287626, 17482416, 17349903, 17571817, 16800117, 17266592, 17312864, 16993089, 16850473, 16761518, 16885609, 16850491, 17292389, 17447285, 17137548, 17274688, 17454645, 16880913, 16854764, 17449539, 17511969, 17347884, 17127217, 17127183, 17445842, 17308279, 16944131, 17160840, 17117680, 17505051, 17393684, 16993083, 17432295, 17274679, 16761474, 17044178, 17627518, 17371756, 16944138, 16609964, 17525539, 16885604, 17157933, 16840115, 16609956, 17407272, 17190202, 17308280, 17190201, 17268950, 16800035, 17268301, 17682485, 17376258, 17065661, 17447279, 17581000, 17303945, 17627521, 18282552, 17403298, 17571825, 16761524, 17517911, 18279108, 18279029, 17374233, 17265369, 17454644, 17495596, 17517886, 17505049, 17580999, 17883256, 17645724, 17142222, 16993087, 17447331, 17525528, 17462882, 17393686, 16993080, 17452158, 16820023, 17453983, 16980627, 17448981, 16980643, 17160842, 17842229, 17303934, 17127223, 17517253, 16609957, 17490282, 17487058, 17965097, 17398508, 17636780, 16993077, 17581103, 17737172, 17505209, 17682502, 17627533, 17169558, 16910307, 18166718, 16854769, 17349836, 16924668, 17490327, 17356478, 17065645, 17253994, 17065694, 16882006, 17137529, 17842230, 17081780, 17581004, 17627523, 17371723, 16910308, 17857237, 17894090, 17137580, 17289445, 16854810, 17830284, 17268328, 17487061, 16885606, 17393687, 18024692, 17477761, 17610953, 17571816, 18015361, 16980618, 17268384, 16731157, 17517264, 17031584, 17700906, 17700925, 17117691, 17965090, 17312861, 17442983, 17347842, 17169479, 16609968, 17445836, 17617032, 18282565, 17303967, 16944143, 17525531, 17559862, 18279106, 17610902, 17610813, 18007921, 17857175, 17160845, 17127189, 16885603, 17857262, 17610871, 17571822, 17674331, 17505054, 17355911, 17559898, 17065640, 17495599, 17360575, 16609973, 17349881, 17517261, 16800068, 17965083, 17627528, 16890613, 17303901, 17700929, 18279027, 17081687, 17308281, 17026560, 17355908, 18166645, 17858975, 17608819, 17805254, 17339109, 17006945, 17491910, 17857298, 16961073, 17965088, 17290056, 17303986, 17312860, 16890617, 17451514, 17169526, 16993090, 17081098, 17894098, 18279035, 17452143, 17303949, 17011122, 17700930, 16882013, 18279113, 18007876, 17167711, 17472216, 17645723, 17517881, 17857291, 16982923, 17355241, 17201993, 16993086, 17201979, 17169490, 16800081, 17137517, 17905194, 17231003, 18007877, 17965091, 17268376, 17274674, 17803194, 17160834, 16697980, 17381373, 17842226, 17645726, 17490314, 17278689, 17398509, 17858980, 17355916, 17147611, 17465607, 17743514, 16993078, 17905150, 17339722, 17434829, 17231022, 17525799, 17026583, 17429890, 17857359, 16891355, 17608835, 17866713, 17691165, 17133971, 17447312, 16980597, 17265366, 16919952, 17268338, 17053281, 17140480, 17429860, 17181707, 16891377, 16879263, 17303980, 17627522, 17347840, 17737175, 16885611, 17447259, 16731160, 17608830, 17268300, 18279023, 17109582, 16980646, 17447306, 17207111, 16627560, 17120991, 17447328, 18007880, 17581001, 17268965, 17120985, 17489529, 17371738, 17816304, 17432299, 17137565, 17127174, 17127179, 17265371, 17449543, 18094932, 17610915, 17581016, 17894094, 17137576, 17447316, 17965086, 16998344, 18279037, 17290567, 18007879, 18024691, 17560965, 17608794, 17525526, 17581101, 17965092, 16993073, 17109574, 17347857, 17403260, 17297272, 17385972, 17120988, 17231004, 17349854, 16993074, 17447342, 17445803, 16998340, 17453970, 17339717, 17356472, 16944136, 17447264, 17447319, 17005539, 18273744, 17559854, 16627561, 17253993, 17268956, 16891423, 17445839, 16961682, 17014984, 17737173, 17610762, 16659284, 17292387, 17610734, 17349875, 16750840, 17127209, 17966714, 17109580, 17816296, 17268973, 16820006, 17312866, 16808479, 16882015, 16924697, 17434760, 17207092, 17268314, 17894089, 17525536, 17308287, 17965093, 17743513, 17581006, 18273719, 16840114, 17393679, 16882014, 17505045, 17023679, 16998287, 16882012, 17403315, 17866724, 17109586, 17371731, 17505060, 17432297, 16890619, 17445837, 16944140, 16609953, 16731158, 17477753, 16885616, 17505050, 16761466, 17465627, 17268374, 16880911, 17429900, 16882010, 16761478, 17518416, 16854828, 17303910, 17448982, 17231028, 17505063, 17510172, 17580998, 18281397, 17024778, 17525525, 17627520, 17517913, 17347877, 17188902, 18279114, 17735158, 17268333, 17482419, 17477744, 17511313, 16919933, 17312862, 17312852, 16879273, 17928049, 17231032, 16731146, 17268386, 17190206, 17517259, 17120989, 17352733, 16919929, 17627534, 17454646, 17201909, 17477748, 16943316, 17608800, 17169546, 17866716, 16731150, 17389345, 16944132, 17190222, 17127220, 16993072, 17449542, 17432298, 17081776, 16980661, 17816333, 17284850, 17393682, 16800103, 17581113, 17137581, 17065659, 17160841, 17349867, 17489508, 17704320, 17268307, 16910310, 17109593, 18167058, 17816293, 16609963, 16609967, 17169494, 17562583, 16980659, 17006931, 16919992, 17894099, 17965084, 17044179, 16731161, 16919981, 17447288, 17627531, 17815749, 18007922, 17157929, 17389407, 17117665, 16849353, 17495598, 17444117, 17636832, 17830282, 18024693, 17965087, 17403268, 16891402, 17127212, 17444128, 17265380, 17292407, 17231010, 16767466, 17445810, 17207129, 17268942, 17628165, 17442982, 17700926, 16853729, 17065702, 17253996, 17445832, 17489523, 17190205, 17184737, 17207102, 17645772, 17207142, 17265379, 17517891, 17505048, 17883265, 18166995, 17349828, 16890620, 17371715, 16609960, 17490296, 16609974, 17885228, 17559844, 17271995, 17816290, 17444118, 17303966, 17830285, 17081099, 17857331, 17137521, 17505059, 17259006, 17964958, 17287625, 17447270, 16944142, 17645725, 17160839, 17355240, 16609976, 17374251, 17610802, 18279022, 18015824, 18282544, 17014982, 17700927, 16890621, 17608821, 17445844, 17355242, 17023686, 17207139, 16821107, 17127188, 17517256, 17608842, 16659285, 17364283, 16919962, 17347875, 17490299, 17105306, 17894091, 17966713, 17449541, 16880927, 17389384, 17355910, 17905192, 17966718, 17894087, 17581009, 17044182, 17006952, 17263135, 17081860, 17965082, 17842223, 17190211, 17127178, 17278691, 17278683, 17831691, 16761534]
        test_tracks = [17081856, 17181697, 17393675, 17393683, 17456150, 16731159, 17601055, 17610786, 16891430, 17691176, 17905193, 17965098, 17495597, 17871918, 17581106, 16889396, 17581110, 17454647, 17477179, 17601087, 17268294, 17169488, 17610838, 17312858, 17268321, 17389154, 17312867, 17389157, 17608806, 17389160, 17389161, 17253995, 17389164, 17389165, 17389166, 16980587, 17389172, 17477749, 17308278, 17966712, 17308284, 17462909, 17477759, 17355919, 16761488, 17268371, 16761494, 17268375, 16980634, 16879262, 17268382, 16609955, 16609962, 17317048, 17231034, 18279107, 17735159, 18279109, 18279110, 17815751, 17014983, 17517257, 17857226, 17259721, 16761550, 17349842, 17023698, 17487060, 17610964, 17434835, 17265368, 17268955, 17044196, 17278699, 18024690, 17349878, 16750841, 16750842, 16750844, 17349887, 16750847, 17127176, 17465616, 17127187, 17559830, 17525537, 17407273, 17157930, 16800041, 17407277, 17407278, 17407280, 17407281, 17407282, 17407284, 17407285, 17407287, 17127227, 17201980, 17559869, 18168637, 18168639, 17407294, 17407297, 18168641, 17208648, 17190218, 17559889, 17807700, 17559893, 16882007, 17207133, 17447265, 16918882, 17490277, 16821103, 17081712, 16821105, 17447295, 17137536, 17137540, 17627525, 17627527, 17627529, 17453967, 17381367, 17490325, 17137560, 17964957, 16998302, 17636773, 16998311, 16919977, 17081784, 17901500, 17803196, 17445831, 17287624, 17581002, 17445834, 16866766, 17581008, 17445841, 17581012, 17581013, 16998356, 17636830, 17015776, 17571818, 17571821, 17571823, 17385967, 17905137, 17905142, 17905143, 17735160, 17117695, 17352732, 17965095, 17428539, 17830283, 17883264, 17465617, 17559835, 17807677, 17447254, 17807711, 17303931, 17490304, 17965096, 17691131, 17449540, 17109594, 16697967, 16609961, 16609971, 17472231, 17081102, 17472314, 16944134, 16944141, 16731147, 17581102, 17581117, 17312854, 17271994, 17349892, 17349900, 17517883, 16840116, 18273745, 17737174, 17026596, 17452781, 16854814, 17207108, 18007875, 16885602, 17571824, 17571826]
    elif ds == 'DSN_100':
        train_tracks = [17610802, 16910310, 16944132, 17065687, 16885608, 17610719, 17393682, 16885604, 17065684, 17444118, 17065674, 17451514, 16885616, 17448975, 16944138, 17371723, 17495598, 17495600, 17253994, 17610762, 17610902, 17289440, 16944134, 16885602, 17442982, 17610924, 16944137, 17610915, 17137581, 16885609, 17065694, 17610746, 17571826, 17444121, 17065661, 17371700, 17448982, 17120985, 17289445, 16885606, 16944131, 17120991, 17434760, 17452164, 16944135, 17393679, 17448981, 17452158, 17610871, 16944136, 17385972, 17120989, 17026560, 17253996, 17610813, 17065659, 16944142, 17253998, 17495599, 17610953, 16944141, 17253992, 17444117, 17147611, 17560965, 16910308, 17253993, 16910307, 17393677, 17385966, 17026583, 17371715, 17065645, 17065702, 17065640, 17444128, 17026596, 16944140, 16885603, 16885611]
        test_tracks = [17610786, 17393675, 17253995, 17495597, 17385967, 17610964, 17610838, 17381367, 17140480, 16944143, 17495596, 17495601, 16885601, 17452143, 17610863, 17452152, 17094053, 17442983, 17120988, 17610734]
    else:
        raise Exception(f"Dataset {ds} invalid, use DSN_100 or DSN_1k")

    return train_tracks, test_tracks

def load_raw_dsn_data(size: Literal['DSN_1k', 'DSN_100', 'DSN_1k_original']) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Load provided raw DSN data, returns tuple of mons and drs

    `mons`: list of tuples (track_id, time-series track)
    `drs`: list of tuples (track_id, discrepancy report information)

    Returns: `tuple[list[pd.DataFrame], list[pd.DataFrame]]`

    Tested (`test_helpers.load_raw_dsn_data`)
    """

    if size not in ['DSN_1k', 'DSN_100', 'DSN_1k_original']:
        raise Exception(f"Dataset size must be \'DSN_1k\', \'DSN_1k_original\' or \'DSN_100\', not {size}")

    if size == 'DSN_1k_original':
        mons_pkl_abs = os.path.join(config.raw_data_dir, size, config.mons_1k_original_pkl_fn)
        drs_pkl_abs = os.path.join(config.raw_data_dir, size, config.drs_1k_original_pkl_fn)
    if size == 'DSN_1k':
        mons_pkl_abs = os.path.join(config.raw_data_dir, size, config.mons_1k_pkl_fn)
        drs_pkl_abs = os.path.join(config.raw_data_dir, size, config.drs_1k_pkl_fn)
    elif size == 'DSN_100':
        mons_pkl_abs = os.path.join(config.raw_data_dir, size, config.mons_100_pkl_fn)
        drs_pkl_abs = os.path.join(config.raw_data_dir, size, config.drs_100_pkl_fn)
    else:
        raise Exception(f"Size {size} not valid, use DSN_100 or DSN_1k.")

    # monitor data time series 
    # Open the file in binary mode
    with open(mons_pkl_abs, 'rb') as file:
        # Load the data from the file
        mons = pickle.load(file)

    # discrepancy reports     
    # Open the file in binary mode
    with open(drs_pkl_abs, 'rb') as file:
        # Load the data from the file
        drs = pickle.load(file)

    return mons, drs

def gen_drs_dict(mons: pd.DataFrame, drs: pd.DataFrame) -> dict[int, int]:
    """
    Generate dictionary mapping track indices to discrepancy report indices
    ie. {track1_idx: [dr1_idx, dr5_idx], track2_idx: [dr2_idx], ...}

    `mons`: Raw DSN time-series file containing parameters and time-steps (pd.DataFrame)  
    `drs`: Raw DSN time-series labeled anomalies (pd.DataFrame)     

    Returns: `dict[int, int]`

    Tested (`test_helpers.gen_drs_dict`)
    """
    ret_dict = dict()
    for drs_idx in range(len(drs)):
        track_id = drs[drs_idx][0]
        for track_idx in range(len(mons)):
            if mons[track_idx][0] == track_id:
                break
        if mons[track_idx][0] != drs[drs_idx][0]:
            raise Exception(f"Found a dr without a corresponding track: dr_id {drs[drs_idx][0]}")
        cur_list = ret_dict.get(track_idx, [])
        cur_list.append(drs_idx)
        ret_dict[track_idx] = cur_list
    return ret_dict

###############################################
# SMD Helper Functions # SMD Helper Functions #
###############################################

def get_SMD_machine_names() -> list[str]:
    """
    List of SMD machine names

    Returns: `list[str]`

    NOT UNIT TESTED  
    Reason: Simple hard-coded list
    """
    dataset_files = ['machine-1-1.txt', 'machine-1-2.txt', 'machine-1-3.txt', 'machine-1-4.txt', 'machine-1-5.txt',
                     'machine-1-6.txt', 'machine-1-7.txt', 'machine-1-8.txt', 'machine-2-1.txt', 'machine-2-2.txt',
                     'machine-2-3.txt', 'machine-2-4.txt', 'machine-2-5.txt', 'machine-2-6.txt', 'machine-2-7.txt',
                     'machine-2-8.txt', 'machine-2-9.txt', 'machine-3-1.txt', 'machine-3-2.txt', 'machine-3-3.txt',
                     'machine-3-4.txt', 'machine-3-5.txt', 'machine-3-6.txt', 'machine-3-7.txt', 'machine-3-8.txt',
                     'machine-3-9.txt', 'machine-3-10.txt', 'machine-3-11.txt']
    return dataset_files

#########################################################
# SMAP/MSL Helper Functions # SMAP/MSL Helper Functions #
#########################################################

def get_chan_split_anom_list(chan_id: str, split: Literal['Train', 'Test', 'Both']) -> list[list[int,int]]:
    """
    Given a channel ID and a split, return the labels

    `chan_id`: SMAP or MSL channel ID (str)  
    `split`: Split for data to return anomaly labels for (str)

    Returns: `list[list[int,int]]`
    
    Tested (`test_helpers.test_get_chan_split_anom_list`)
    """
    # Input check
    if split not in ['Train', 'Test', 'Both']:
        raise Exception(f"Split {split} is not valid, use Train, Test, or Both.")

    dataset = get_sc_from_chan_id(chan_id)
    labels_path = os.path.join(config.val_dict[f'{dataset.lower()}_data_dir'], split, 'Labels', f'{chan_id}.l.pkl')
    with open(labels_path, 'rb') as file:
        labels = pickle.load(file)
    return labels

def get_smap_msl_file_names() -> list[str]:
    """
    Return hard-coded file names of SMAP and MSL files

    Returns: `list[str]`

    NOT UNIT TESTED  
    Reason: Simple hard-coded list
    """
    # Per https://github.com/khundman/telemanom/issues/8
    # we ignore T-10
    test_files = ['E-8', 'P-10', 'A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8', 'A-9', 'B-1', 'C-1', 'C-2', 'D-1', 'D-11', 'D-12', 'D-13', 'D-14', 'D-15', 'D-16', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'E-1', 'E-10', 'E-11', 'E-12', 'E-13', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-9', 'F-1', 'F-2', 'F-3', 'F-4', 'F-5', 'F-7', 'F-8', 'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 'G-7', 'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'P-1', 'P-11', 'P-14', 'P-15', 'P-2', 'P-3', 'P-4', 'P-7', 'R-1', 'S-1', 'S-2', 'T-1', 'T-12', 'T-13', 'T-2', 'T-3', 'T-4', 'T-5', 'T-8', 'T-9']

    return test_files

def get_smap_msl_train_file_names() -> list[str]:
    """
    Returns a list of all file names for training data for both MSL and SMAP

    Returns: `list[str]`
    
    NOT UNIT TESTED  
    Reason: Simple hard-coded list
    """
    return get_smap_msl_file_names()

def get_smap_msl_test_file_names() -> list[str]:
    """
    Returns a list of all file names for testing data for both MSL and SMAP

    Returns: `list[str]`
    
    NOT UNIT TESTED  
    Reason: Simple hard-coded list
    """
    return get_smap_msl_file_names()

def get_all_channels() -> list[str]:
    """
    Returns a list of all channels for both MSL and SMAP

    Returns: `list[str]`
    
    NOT UNIT TESTED  
    Reason: Simple hard-coded list
    """
    return get_smap_msl_test_file_names()

def get_sc_channels(sc: Literal['MSL', 'SMAP']) -> list[str]:
    """
    Returns a list of file names for specified spacecraft

    `sc`: Spacecraft string name (str)

    Returns: `list[str]`
    
    Tested (`test_helpers.test_get_sc_channels`)
    """
    # Input check
    if sc not in ['MSL', 'SMAP']:
        raise Exception(f"Spacecraft {sc} is not valid, use MSL or SMAP.")

    data_path = os.path.join(config.val_dict[f'{sc.lower()}_data_dir'], 'Train')

    # Load data
    track_names = [path.split('.')[0] for path in os.listdir(data_path)]

    return track_names

def get_all_chan_id_from_sc(sc: Literal['MSL', 'SMAP']) -> list:
    """
    Return a list of channel IDs associated with the spacecraft.
    
    `sc`: Spacecraft string name (str)

    Returns: `list`
    
    Tested (`test_helpers.get_all_chan_id_from_sc`)
    """
    if sc not in ['MSL', 'SMAP']:
        raise Exception(f"Spacecraft {sc} is not valid, use MSL or SMAP.")

    if sc == 'MSL':
        sc_list = ['M-6','M-1','M-2','S-2','P-10','T-4','T-5','F-7','M-3','M-4','M-5','P-15','C-1','C-2','T-12','T-13','F-4','F-5','D-14','T-9','P-14','T-8','P-11','D-15','D-16','M-7','F-8']
    if sc == 'SMAP':
        sc_list = ['P-1','S-1','E-1','E-2','E-3','E-4','E-5','E-6','E-7','E-8','E-9','E-10','E-11','E-12','E-13','A-1','D-1','P-2','P-3','D-2','D-3','D-4','A-2','A-3','A-4','G-1','G-2','D-5','D-6','D-7','F-1','P-4','G-3','T-1','T-2','D-8','D-9','F-2','G-4','T-3','D-11','D-12','B-1','G-6','G-7','P-7','R-1','A-5','A-6','A-7','D-13','P-2','A-8','A-9','F-3']
    return sc_list

def get_sc_from_chan_id(chan_id: str) -> Literal['MSL', 'SMAP']:
    """
    Return the string for the spacecraft associated with the channel ID.

    `chan_id`: Channel id to search for the name of spacecraft (str)
    
    Returns: `Literal['MSL', 'SMAP']`
    
    Tested (`test_helpers.get_all_chan_id_from_sc`)
    """
    sc_list_msl = get_all_chan_id_from_sc('MSL')
    sc_list_smap = get_all_chan_id_from_sc('SMAP')

    if chan_id in sc_list_msl:
        return 'MSL'
    elif chan_id in sc_list_smap:
        return 'SMAP'
    else:
        raise Exception(f"Invalid channel ID: {chan_id}")

def get_split_from_chan_id(chan_id: str, split: Literal['Train', 'Test']) -> np.ndarray:
    """
    Return either the original training or testing numpy array for the specified channel ID

    `chan_id`: Channel id to get split data for (str)  
    `split`: Split to collect data from (str)  

    Returns: `np.ndarray`
    
    Tested (`test_helpers.get_split_from_chan_id`)
    """
    # Input check
    if split not in ['Train', 'Test']:
        raise Exception(f"Split {split} is not valid, use Train or Test.")

    sc = get_sc_from_chan_id(chan_id)
    data_path = os.path.join(config.val_dict[f'{sc.lower()}_data_dir'], split, 'Tracks', f"{chan_id}.df.pkl")
    with open(data_path, 'rb') as file:
        arr_np = pickle.load(file).to_numpy()
    return arr_np

###################################
# TranAD Helpers # TranAD Helpers #
###################################

def convert_anoms_to_tranad(anoms: Union[list[list[int, int]], np.ndarray], out_shape: tuple[int, int]) -> np.ndarray:
    """
    Given an (n,d) shape `out_shape`, return a float numpy array with all zeros except for time steps where anomalies are
    defined by `anoms`, all dimensions at those time steps are set to 1 inclusive.

    `anoms`: List of anomalies as list of start and end indices (`[[start1,end1],[start2,end2],...]`)
    `out_shape: (n,d) numpy shape where the `n` values are replaced by `1` across `d` if they are included as an anomaly in `anoms` and `0` otherwise

    Returns: `np.ndarray`

    Tested (`test_helpers.convert_anoms_to_tranad`)
    """
    anoms = np.asarray(anoms)
    arr_out = np.zeros(out_shape)
    for anom in anoms:
        arr_out[anom[0]:anom[1]+1,:] = 1.
    return arr_out
    
def convert_anoms_from_tranad(anoms: np.ndarray) -> list[list[int, int]]:
    """
    Given a float numpy array shaped (n,d) with all zeros except for time steps where anomalies are defined
    (1 at those time steps across all dimensions), return a list of ranges (inclusive) where anomalies are defined.

    `anoms: (n,d) numpy shape where the `n` values are marked as `1` across `d` if they are included as an anomaly in `anoms` and `0` otherwise

    Returns: `list[list[int, int]]`

    Tested (`test_helpers.convert_anoms_from_tranad`)
    """
    anoms_out = list()
    intra_anom = False # Flag to see if we're currently in an anomaly
    start_idx = 0
    end_idx = 0
    for i in range(anoms.shape[0]):
        if intra_anom:
            # Exited anomaly range
            if anoms[i][0] == 0.:
                intra_anom = False
                anoms_out.append([start_idx, end_idx])
            # Within anomaly range
            elif anoms[i][0] == 1.:
                # End of track
                if i == anoms.shape[0] - 1:
                    anoms_out.append([start_idx, i])
                # Not end of track
                else:
                    end_idx = i
        else:
            # Not within anomaly range
            if anoms[i][0] == 1.:
                start_idx = i
                end_idx = i
                intra_anom = True
    return anoms_out
