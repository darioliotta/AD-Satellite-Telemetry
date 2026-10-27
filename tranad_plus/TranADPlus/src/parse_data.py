###############################
# Imports # Imports # Imports #
###############################

# System
import os
import pickle
from typing import Literal

# Packages
import numpy as np
from tqdm import tqdm
from TranADPlus.src.helpers import flatten_list, make_dir_tree, get_dataset_split_absolute_file_paths, load_npy_from_path, load_df_from_path, get_dataset_split_track_names
import TranADPlus.src.global_config as global_config

# Load config
config = global_config.config

#######################################################
# Preprocess Data # Preprocess Data # Preprocess Data #
#######################################################

def parse_tracks(dataset: Literal['MSL', 'SMAP', 'DSN_100', 'DSN_1k', 'SWaT', 'WADI', 'SMD'] = None) -> None:
    """
    Parse specified datasets and obtain valuable EDA information
    Saves to config's `parse_save_dir`: markdown and pickle files

    Markdown (dir: `md`):
    - `byte.md`
        - Parameter numbers that are bytes
    - `float64.md`
        - Parameter numbers that are float64
    - `nans.md`
        - Number of nans per parameter per track
    - `params.md`
        - Number of unique parameters
    - `shapes.md`
        - Unique track shapes
    - `types.md`
        - unique track types
    - `byte_vals.md`
        - unique byte values for the dataset

    Pickle (dir: `pkl`):
    - `num_nans.pkl`
        - Number of nans per parameter per track. Shape: (track #, parameter #)
    - `shapes.pkl`
        - Number of shapes per track. Shape: (track #)
    - `params.pkl`
        - Number of parameters per track. Shape: (track #)
    - `types.pkl`
        - Parameter type per parameter per track. Shape: (track #, parameter #)
    - `byte_vals.pkl`
        - Unique byte values per parameter. Shape: (parameter # x byte_vals x # (variable))

    Sanity checks (exception raised if not)
    1) Each parameter has the same type for each parameter across tracks
    2) Each track has the same number of parameters
    3) Therefore, each track has the same types and number of parameters

    Observations:
    - Each track may have a different number of time series values
    DSN:
    - There are two types of values: float64 and bytes
    SMAP/MSL:
    - Only float64 type

    Returns: `None`

    Tested (`test_parse_data.test_parse_tracks`)
    """
    # Input check
    if dataset not in ['MSL', 'SMAP', 'DSN_100', 'DSN_1k', 'SWaT', 'WADI', 'SMD']:
        raise Exception(f"Dataset {dataset} is not valid, use MSL, SMAP, DSN_100, 'DSN_1k', 'SMD', WADI, or 'SWaT.")

    # Generic set-up
    parse_save_dir = config.val_dict['parse_save_dir']
    parse_sub_dir = os.path.join(parse_save_dir, dataset)
    # Make directory tree
    make_dir_tree([parse_save_dir, parse_sub_dir])

    splits = ['Train', 'Test', 'Both']

    for split in splits:
        # Save directory 
        parse_split_dir = os.path.join(parse_sub_dir, split)
        parse_pkl_dir = os.path.join(parse_split_dir, 'pkl')
        parse_md_dir = os.path.join(parse_split_dir, 'md')
        # Make directory tree
        make_dir_tree([parse_split_dir, parse_pkl_dir, parse_md_dir])

        # Get track names
        absolute_track_names = get_dataset_split_absolute_file_paths(dataset, split)
        track_names = get_dataset_split_track_names(dataset, split)
        
        # Get parameters number by loading a single track
        track_np = load_npy_from_path(absolute_track_names[0])

        # Constants (n_params populated at first hit)
        N_PARAMS = track_np.shape[1]
        N_TRACKS = len(track_names)
        print(f"Parsing {N_TRACKS} tracks with {N_PARAMS} parameters.")

        # Values we're obtaining
        shapes_l = [] # Store number of shapes per track
        params_l = [] # Store number of parameters per track
        types_l = [] # Store parameter type per parameter per track
        unique_types_l = [] # Store unique parameter types, not pickled, only used for making md file
        num_nans_l = [[] for i in range(N_TRACKS)] # Store number of nans per parameter per track
        byte_vals_l = [[] for i in range(N_PARAMS)] # Store unique byte values encountered per parameter

        # Files we're saving to
        shapes_fn = "shapes.md"
        params_fn = "params.md"
        types_fn = "types.md"
        nans_fn = "nans.md"
        byte_vals_fn = "byte_vals.md"
        float64_fn = "float64.md"
        byte_fn = "byte.md"

        # Open files
        shapes_file = open(os.path.join(parse_md_dir, shapes_fn), 'w')
        params_file = open(os.path.join(parse_md_dir, params_fn), 'w')
        types_file = open(os.path.join(parse_md_dir, types_fn), 'w')
        nans_file = open(os.path.join(parse_md_dir, nans_fn), 'w')
        byte_vals_file = open(os.path.join(parse_md_dir, byte_vals_fn), 'w')
        float64_file = open(os.path.join(parse_md_dir, float64_fn), 'w')
        byte_file = open(os.path.join(parse_md_dir, byte_fn), 'w')

        # Write header per file
        shapes_file.write("track | unique shape\n")
        shapes_file.write("--- | ---\n")

        params_file.write("track | unique number of parameters\n")
        params_file.write("--- | ---\n")

        types_file.write("track | parameter | unique type\n")
        types_file.write("--- | --- | ---\n")

        nans_file.write("track | parameter | description\n")
        nans_file.write("--- | --- | ---\n")

        byte_vals_file.write("track | unique byte value\n")
        byte_vals_file.write("--- | ---\n")

        float64_file.write("| parameter |\n")
        float64_file.write("| --- |\n")

        byte_file.write("| parameter |\n")
        byte_file.write("| --- |\n")

        # Iterate over tracks to obtain data
        for i in tqdm(range(len(track_names)), desc="Iterating over tracks"):
            # Get track name
            track_name = track_names[i]

            # Get track
            track = load_df_from_path(absolute_track_names[i])

            # Get shape
            track_shape = track.shape
            if track_shape not in shapes_l:
                shapes_file.write(f"{track_name:<4} | {track_shape}\n")
            shapes_l.append(track_shape)

            # Get number of parameters
            track_params = track_shape[1]
            if track_params not in params_l:
                params_file.write(f"{track_name:<4} | {track_params}\n")
            params_l.append(track_params)

            # Get number of timesteps
            track_data = track_shape[0]

            # Sanity check: each subsequent track should have the same number of parameters
            if track_params != N_PARAMS:
                raise Exception(f"ERROR: track i={i} ({track_name}) has {track_params} parameters but {N_PARAMS} was expected")
                continue

            # Get number of nans per columns
            num_nans = track.isna().sum()

            # Get parameter types per column (Pandas ensures each element per column is the same type)
            param_types = track.dtypes

            # Update types list
            types_l.append(list(param_types))
            # Sanity check: each subsequent track should have the same types for the same parameters
            if i > 0 and types_l[i] != types_l[0]:
                raise Exception(f"ERROR: track {i} ({track_name}) ({track_names[i]}) has different parameter types than track 0.")
            
            # Identify unique byte values per byte parameter
            # Per parameter
            for j in range(track_params):
                # Get parameter type
                p_type = param_types.iloc[j]
                # Write to file is new param type encountered
                if p_type not in unique_types_l:
                    types_file.write(f"{track_name} | {j} | {p_type}\n")
                    unique_types_l.append(p_type)
                if p_type == np.dtype('float64'):
                    # Write to file for first track
                    if i == 0:
                        float64_file.write(f"| {j} |\n")
                    if num_nans.iloc[j] == 0:
                        custom_str = "no"
                        desc = f"{custom_str:>13} nan"
                        nans_file.write(f"{track_name:<4} | {j:<4} | {desc}\n")
                    elif num_nans.iloc[j] == track_data:
                        custom_str = "all"
                        desc = f"{custom_str:>13} nan"
                        nans_file.write(f"{track_name:<4} | {j:<4} | {desc}\n")
                    else:
                        desc = f"{num_nans.iloc[j]:>6}/{track_data:<6} nan"
                        nans_file.write(f"{track_name:<4} | {j:<4} | {desc}\n")
                elif p_type == np.dtype('O'):
                    # Write to file for first track
                    if i == 0:
                        byte_file.write(f"| {j} |\n")
                    # Get unique byte values
                    byte_vals = list(track.iloc[:,j].unique())
                    for byte_val in byte_vals:
                        # Write to file if entirely new byte value encountered
                        if byte_val not in flatten_list(byte_vals_l):
                            byte_vals_file.write(f"{track_name:<4} | {byte_val}\n")
                        # Append to unique list for this track if new value encountered
                        if byte_val not in byte_vals_l[j]:
                            byte_vals_l[j].append(byte_val)
                else:
                    raise Exception(f"ERROR: track {i} ({track_name}) has new parameter type of {p_type}.")
                num_nans_l[i].append(num_nans.iloc[j])

        # Close files
        shapes_file.close()
        params_file.close()
        types_file.close()
        nans_file.close()
        byte_vals_file.close()
        float64_file.close()
        byte_file.close()

        # Write collected variables as pickle files
        with open(os.path.join(parse_pkl_dir, f'shapes.pkl'), 'wb') as shapes_file:
            pickle.dump(shapes_l, shapes_file)
        with open(os.path.join(parse_pkl_dir, f'params.pkl'), 'wb') as params_file:
            pickle.dump(params_l, params_file)
        with open(os.path.join(parse_pkl_dir, f'types.pkl'), 'wb') as types_file:
            pickle.dump(types_l, types_file)
        with open(os.path.join(parse_pkl_dir, f'num_nans.pkl'), 'wb') as nans_file:
            pickle.dump(num_nans_l, nans_file)
        with open(os.path.join(parse_pkl_dir, f'byte_vals.pkl'), 'wb') as byte_vals_file:
            pickle.dump(byte_vals_l, byte_vals_file)

        print(f"Wrote to {os.path.join(parse_split_dir)}")
    return None

