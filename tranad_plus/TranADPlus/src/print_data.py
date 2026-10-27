###############################
# Imports # Imports # Imports #
###############################

# System
import os
import pickle
from typing import Literal

# Packages
import numpy as np
import pandas as pd
from datetime import datetime

from TranADPlus.src.helpers import \
    convert_str_to_int_list, load_raw_dsn_data,\
    get_split_from_chan_id, get_dataset_split_absolute_file_paths

import TranADPlus.src.global_config as global_config

# Load config
config = global_config.config

#############################################################
# Generic functions # Generic Functions # Generic Functions #
#############################################################

def print_dataset_anomaly_info(ds:Literal['WADI', 'SWaT', 'SMAP', 'DSN_1k', 'DSN_100', 'SWaT', 'SMD', 'MSL', 'WADI'], split:Literal['Train', 'Test', 'Both']):
    """
    For dataset, print statistics on the anomalous time-steps.

    `dataset`: Which dataset to print length stats for (str)  
    `split`: Which split to print length stats for (str)  

    Returns: `None`
    
    Tested (`test_print_data.print_dataset_anomaly_info`)
    """
    # Input check
    if ds not in ['MSL', 'SMAP', 'SWaT', 'SMD', 'MSL_SMAP', 'DSN_1k', 'DSN_100', 'WADI']:
        raise Exception(f"Dataset {ds} is not valid, use MSL, SMAP, MSL_SMAP, WADI, DSN_100 or DSN_1k.")

    # Get track lengths
    if ds in ['MSL', 'SMD', 'SMAP', 'WADI', 'SWaT', 'DSN_1k', 'DSN_100', 'WADI']:
        with open(os.path.join(config.parse_save_dir, ds, split, 'pkl', f'shapes.pkl'), 'rb') as shapes_file:
            shapes = np.asarray(pickle.load(shapes_file))

    total_len = np.sum(shapes[:,0])

    # Get number of anomalies
    label_dir=os.path.join(config.original_data_dir, ds, split, 'Labels')
    num_anoms_l = list()
    
    filenames = os.listdir(label_dir)
    filenames.sort()
    for filename in filenames:
        A = np.load(os.path.join(label_dir, filename), allow_pickle=True)
        anomalies_len = 0
        for (start_idx, end_idx) in A:
            anomalies_len = end_idx +1 - start_idx
            num_anoms_l.append(anomalies_len)

    num_anoms = np.sum(num_anoms_l)

    # Print info
    print(pd.DataFrame(num_anoms_l,columns=[f"{ds}_{split}"]).describe())
    print(f"{split} len {ds}: {total_len} ({num_anoms} anomalies: {num_anoms/total_len*100:0.2f}% is anomalous)")

def print_hyperparameters(hp_dict: dict[str, str]) -> None:
    """
    Print given hyperparameters

    `hp_dict`: Hyperparameters dictionary to print key and values for

    Returns: `None`
    
    Tested (`test_print_data.print_hyperparameters`)
    """
    for key in hp_dict.keys():
        print(f"{key}: {hp_dict[key]}")

    return None

def print_dataset_nan_info(dataset: Literal['MSL', 'SMAP', 'DSN_1k', 'DSN_100', 'SWaT', 'SMD', 'WADI'], split: Literal['Both', 'Train', 'Test'] = 'Both') -> None:
    """
    Print nan info for each parameter for the given dataset

    `dataset`: Which dataset to print length stats for (str)  
    `split`: Which split to print length stats for (str)  

    Returns: `None`

    Tested (`test_print_data.print_dataset_nan_info`)
    """
    # Input check
    if dataset not in ['MSL', 'SMAP', 'DSN_1k', 'DSN_100', 'SWaT', 'SMD', 'WADI']:
        raise Exception(f"Dataset {dataset} is not valid, use MSL, SMAP, SWaT, SMD, WADI, DSN_100, or DSN_1k.")
    pkl_dir = os.path.join(config.parse_save_dir, dataset, split, 'pkl')

    # Load pickles
    with open(os.path.join(pkl_dir, 'num_nans.pkl'), 'rb') as nans_file:
        num_nans = np.asarray(pickle.load(nans_file)) # Stores nans per parameter per track

    with open(os.path.join(pkl_dir, 'shapes.pkl'), 'rb') as shapes_file:
        shapes = np.asarray(pickle.load(shapes_file)) # Stores track shapes

    with open(os.path.join(pkl_dir, 'types.pkl'), 'rb') as types_file:
        types = np.asarray(pickle.load(types_file)) # Stores track types

    nans_per_parameter = np.sum(num_nans,axis=0) # Get nans per parameter across all tracks
    total_values = np.sum(shapes[:,0]) # Get total number of values per parameter (total time series steps)
    print("total_nans:", np.sum(num_nans), f"({np.sum(num_nans)/(total_values*shapes[0,1])*100}%)")

    no_nans = []
    yes_nans = []
    all_nans = []
    some_nans = []
    byte_params = []
    float_params = []

    for i in range(nans_per_parameter.shape[0]):
        # Check if parameter is float64
        if types[0][i] != np.float64:
            byte_params.append(i)
            continue
        float_params.append(i)
        nans = nans_per_parameter[i]
        if nans != 0:
            yes_nans.append(i)
            if nans == total_values:
                all_nans.append(i)
            else:
                some_nans.append(i)
        else:
            no_nans.append(i)
    print(f"Parameters without any nans ({len(no_nans)}):")
    print(no_nans)
    print(f"Parameters with some nans ({len(some_nans)}):")
    print(some_nans)
    print(f"Parameters with all nans ({len(all_nans)}):")
    print(all_nans)
    print(f"Byte parameters ({len(byte_params)}):")
    print(byte_params)
    print(f"Float parameters ({len(float_params)}):")
    print(float_params)
    print(f"Total parameters ({shapes[0][1]})")

    total_params = len(no_nans)+len(some_nans)+len(all_nans)+len(byte_params)
    if (total_params) != shapes[0][1]:
        raise Exception(f"Error: parameter groups don't add up to {shapes[0][1]}, but add up to {total_params}")

def print_dataset_length_stats(dataset: Literal['MSL', 'SMAP', 'MSL_SMAP' 'DSN_1k', 'DSN_100', 'WADI', 'SWaT'],\
                               split: Literal['Both', 'Train', 'Test'] = 'Both') -> None:
    """
    For dataset, print statistics on the data length.

    `dataset`: Which dataset to print length stats for (str)  
    `split`: Which split to print length stats for (str)  

    Returns: `None`
    
    Tested (`test_print_data.print_dataset_length_stats`)
    """
    # Input check
    if dataset not in ['MSL', 'SMAP', 'MSL_SMAP', 'DSN_1k', 'DSN_100', 'WADI', 'SWaT', 'SMD']:
        raise Exception(f"Dataset {dataset} is not valid, use MSL, SMAP, MSL_SMAP, WADI, SWaT, SMD, DSN_100 or DSN_1k.")

    parse_sub_dir = config.parse_save_dir

    if dataset in ['MSL', 'SMAP', 'DSN_1k', 'DSN_100', 'WADI', 'SWaT', 'SMD']:
        with open(os.path.join(parse_sub_dir, dataset, split, 'pkl', f'shapes.pkl'), 'rb') as shapes_file:
            shapes = np.asarray(pickle.load(shapes_file))
    elif dataset in ['MSL_SMAP']:
        # Combine MSL and SMAP shapes
        with open(os.path.join(parse_sub_dir, 'MSL', split, 'pkl', f'shapes.pkl'), 'rb') as shapes_file:
            shapes_1 = np.asarray(pickle.load(shapes_file))
        with open(os.path.join(parse_sub_dir, 'SMAP', split, 'pkl', f'shapes.pkl'), 'rb') as shapes_file:
            shapes_2 = np.asarray(pickle.load(shapes_file))
        shapes = np.concatenate([shapes_1, shapes_2])
        
    print(pd.DataFrame(shapes[:,0],columns=[f"{dataset}_{split}"]).describe())
    print(f"Total data points: {np.sum(shapes[:,0])}")

def print_prec_rec(tp: int, fp: int, fn: int, padding: int = 4) -> None:
    """
    Given true positives, false positives, and false negatives, print precision, recall, F0.5, and F1.0.

    `tp`: True positives (int)  
    `fp`: False positives (int)  
    `padding`: Print padding (int)  

    Returns: `None`
    
    Tested (`test_print_data.print_prec_rec`)
    """
    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    f = lambda beta: (1. + beta**2) * ( (precision*recall) / ( ((beta**2)*precision) + recall )) 
    padded_str = ' '*padding
    print(f'{padded_str}Precision: {precision:.2f}')
    print(f'{padded_str}Recall: {recall:.2f}')
    print(f'{padded_str}F0.5: {f(0.5):.2f}')
    print(f'{padded_str}F1.0: {f(1.0):.2f}')

#########################################################
# SMAP/MSL Helper Functions # SMAP/MSL Helper Functions #
#########################################################

def print_chan_split_shape(chan_id: str, split: Literal['Train', 'Test', 'Both']) -> None:
    """
    Print the shape of the specified track and split (train, test, both)

    `chan_id`: Which channel ID to get shape for (str)   
    `split`: Which split to get shape for (str)  

    Returns: `None`
    
    Tested (`test_print_data.print_chan_split_shape`)
    """

    # Load data efficiently
    if split == 'Train' or split == 'Both':
        train_np = get_split_from_chan_id(chan_id, 'Train')
        total_np = train_np
    if split == 'Test' or split == 'Both':
        test_np = get_split_from_chan_id(chan_id, 'Test')
        total_np = test_np
    if split == 'Both':
        total_np = np.vstack([train_np,test_np])

    # Print data
    print(f"Shape of channel \"{chan_id}\" split \"{split}\" is {total_np.shape}")
    return None

###############################################
# DSN Helper Functions # DSN Helper Functions #
###############################################

def print_dsn_timestamp_range(size: Literal['DSN_100', 'DSN_1k']) -> None:
    """
    Assuming raw DSN data is available, print the earliest and latest timestamp of the time-series data

    `size`: Which split of DSN to use (str)

    Returns: `None`
    
    Tested (`test_print_data.print_dsn_timestamp_range`)
    """
    earliest = np.inf
    latest = -1*np.inf

    if size not in ['DSN_100', 'DSN_1k']:
        raise Exception(f"DSN size must be either \'DSN_100\' \'DSN_1k\', not {size}")

    mons, drs = load_raw_dsn_data(size)

    for i in range(len(mons)):
        # Convert a 9-digit timestamp to a date
        timestamp = mons[i][1].iloc[0,0] 
        if timestamp < earliest:
            earliest = timestamp

        timestamp = mons[i][1].iloc[-1,0]
        if timestamp > latest:
            latest = timestamp

    date = datetime.utcfromtimestamp(earliest).strftime('%Y-%m-%d %H:%M:%S')
    print("earliest: ", date)

    date = datetime.utcfromtimestamp(latest).strftime('%Y-%m-%d %H:%M:%S')
    print("latest: ", date)

    return None