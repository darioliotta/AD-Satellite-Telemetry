###############################
# Imports # Imports # Imports #
###############################

# System
import os
import pickle
from typing import Literal, Any

# Packages
from sklearn import preprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import TranADPlus.src.helpers as helpers
from TranADPlus.src.helpers import convert_str_to_int_list,\
    load_raw_dsn_data, make_dir_tree, shift_anoms,\
    get_dataset_split_track_names, \
    get_all_channels, gen_drs_dict, get_sc_from_chan_id,\
    replace_byte_constant, replace_byte_dict, replace_byte_one_hot, convert_anoms_from_tranad
import TranADPlus.src.global_config as global_config

# Load config
config = global_config.config

##################################################################
# Parse Data # Parse Data # Parse Data # Parse Data # Parse Data #
##################################################################

def combine_train_test_data(ds: Literal['SMAP', 'MSL', 'SWaT', 'WADI', 'SMD']) -> None:
    """
    Combine training and testing tracks to make a single track for datasets where a single track is split into an initial testing and later training section
    For example, combine train/test for each track in MSL/SMAP into a single A-1 track where train is concatenated with test.
    Outputs to 'Both' directory in `Datasets/Both` directory (pre-preprocessing, post converting to data frames)

    `ds`: Dataset to combine training and testing data for (str)  

    Returns: `None`
    
    Tested (`test_preprocess_data.combine_train_test_data`)
    """
    if ds not in ['SMAP', 'MSL', 'SWaT', 'WADI', 'SMD']:
        raise Exception(f"Dataset {ds} not valid, use SMAP, MSL, SWaT, WADI or SMD.")
    path = os.path.join(config.val_dict[f'{ds.lower()}_data_dir'], 'Train', 'Tracks')
    filenames = [file.split('.')[0] for file in os.listdir(path)]
    for fn in filenames:
        with open(os.path.join(config.val_dict[f'{ds.lower()}_data_dir'], 'Train', 'Tracks', f'{fn}.df.pkl'), 'rb') as file:
            train_np = pickle.load(file).to_numpy()
        with open(os.path.join(config.val_dict[f'{ds.lower()}_data_dir'], 'Test', 'Tracks', f'{fn}.df.pkl'), 'rb') as file:
            test_np = pickle.load(file).to_numpy()
        both_np = np.vstack([train_np, test_np])
        data_dir = os.path.join(config.val_dict[f'{ds.lower()}_data_dir'], 'Both', 'Tracks')
        make_dir_tree(data_dir)
        with open(os.path.join(data_dir, f'{fn}.df.pkl'), 'wb') as file:
            pickle.dump(pd.DataFrame(both_np), file)
    return None

def preprocess_submodule(ds: Literal['SMAP', 'MSL', 'SWaT', 'WADI', 'SMD', 'DSN_100', 'DSN_1k'],\
                         submodule: Literal['OmniAnomaly', 'TranAD', 'MTAD-GAT']) -> None:
    """
    Combine all tracks from the testing split and all tracks in the training split into a single track for each split

    Output to specified submodule

    Only works on preprocessed data

    `ds`: Dataset to serve to the submodule (str)  
    `submodule`: The submodule to serve the dataset (str)  

    Returns: `None`

    NOT UNIT TESTED  
    Reason: This function's purpose is to take preprocessed data and serve it to the specified submodule for use in training. This was used for data collection.
    """
    # Input check
    if ds not in ['SMAP', 'MSL', 'SWaT', 'SMD', 'WADI', 'DSN_100', 'DSN_1k']:
        raise Exception(f"Dataset {ds} not valid, use SMAP, MSL, SWaT, WADI, or SMD.")
    if submodule not in ['OmniAnomaly', 'TranAD', 'MTAD-GAT']:
        raise Exception(f"Submodule {submodule} not valid, use OmniAnomaly, TranAD, or MTAD-GAT.")

    # Create output directory
    if submodule in ['TranAD']:
        output_dir = os.path.join(config.TranAD_dir, 'processed', f"{config.package_name}_{ds}")
    elif submodule in ['MTAD-GAT']:
        output_dir = os.path.join(config.MTAD_GAT_dir, 'datasets', f"{config.package_name}_{ds}", 'processed')
    elif submodule in ['OmniAnomaly']:
        output_dir = os.path.join(config.OmniAnomaly_dir, 'processed')
    else:
        raise Exception(f"Submodule {submodule} not valid, use OmniAnomaly, TranAD, or MTAD-GAT.")
    helpers.make_dir_tree(output_dir)

    # Generate output
    for split in ['Train', 'Test']:
        for track_type in ['Tracks', 'Labels']:
            # Get track/label as a single numpy array
            track_dir = os.path.join(config.preprocess_save_dir, ds, split, track_type)
            track_names = os.listdir(track_dir)
            track_paths = [os.path.join(track_dir, track_name) for track_name in track_names]
            big_track = helpers.merge_tracks(track_paths, window_sz=1, padding=False)

            # Create output based on submodule
            if submodule in ['TranAD']:
                if track_type == 'Labels':
                    if split == 'Train':
                        # TranAD doesn't care about training labels
                        continue
                    output_fn = os.path.join(output_dir, f'labels.npy')
                else:
                    # Scale data
                    if split == 'Train':
                        scaler = preprocessing.MinMaxScaler()
                        big_track = scaler.fit_transform(big_track)
                    elif split == 'Test':
                        big_track = scaler.transform(big_track)
                    output_fn = os.path.join(output_dir, f'{split.lower()}.npy')
                np.save(output_fn, big_track)

            elif submodule in ['MTAD-GAT', 'OmniAnomaly']:
                if track_type == 'Labels':
                    # Labels should be a 1D list stating whether
                    # timestep i is anomalous
                    big_track = np.mean(big_track, axis=1)
                    big_track = big_track.astype(bool)
                    output = os.path.join(output_dir, f"{config.package_name}_{ds}_{split.lower()}_label.pkl")
                else:
                    # Scale data
                    if split == 'Train':
                        scaler = preprocessing.MinMaxScaler()
                        big_track = scaler.fit_transform(big_track)
                    elif split == 'Test':
                        big_track = scaler.transform(big_track)
                    output = os.path.join(output_dir, f"{config.package_name}_{ds}_{split.lower()}.pkl")
                with open(output, 'wb') as file:
                    pickle.dump(big_track, file)

            else:
                raise Exception(f"Submodule {submodule} not valid, use OmniAnomaly, TranAD, or MTAD-GAT.")
    return None

def initialize_dataset(dataset: Literal['SMAP', 'MSL', 'SWaT', 'SMD', 'WADI' ,'DSN_100', 'DSN_1k']) -> None:
    """
    Helper function to call specific dataset initialization subroutines. This takes in a raw dataset in `Datasets/Raw` and outputs them in `Datasets/Original` for parsing with `preprocess_data.preprocess_data`. This function takes the datasets of different formats and makes them a standard format used in the rest of the code.

    The standard format is to take the raw dataset and to produce training and testing tracks in the `Datasets/Original` directory. For each train/test split, there are `Labels` (a list of anomalous region indices like `[[start1,end1],[start2,end2],...]`) and a dataframe of floating and/or binary values containing the time-series. Furthermore, a `Both` split is created where tracks of the same name are concatenated and their anomalies are combined.

    `dataset`: String for the dataset to initialize (str)  

    Returns: `None`

    NOT UNIT TESTED  
    Reason: This function just passes initialization down to a specific initialization function.
    """

    if dataset in ['SMAP', 'MSL']:
        initialize_smap_msl_data_all()
    elif dataset in ['DSN_100', 'DSN_1k']:
        initialize_dsn_data(dataset)
    elif dataset in ['SWaT']:
        initialize_swat_data()
    elif dataset in ['SMD']:
        initialize_smd_data()
    elif dataset in ['WADI']:
        initialize_wadi_data()
    else:
        raise Exception(f"Dataset {dataset} not valid, use SMAP, MSL, SWaT, WADI, SMD, DSN_100, or DSN_1k")
        
    return None

def initialize_wadi_data() -> None:
    """
    Initialize WADI data

    Takes the train and test dataframes and converts them to floating point
    Also converts anomalies to list of indices for anomalies

    Returns: `None`

    NOT UNIT TESTED  
    Reason: This functino initializes the WADI dataset for use in the rest of the code. Unit testing this would mean doing the actual preprocessing and what not, which is better served as something to be done outside of testing.
    """
    # Load data
    wadi_data_abs = os.path.join(config.raw_data_dir, 'WADI', config.wadi_data_fn)
    wadi_label_abs = os.path.join(config.raw_data_dir, 'WADI', config.wadi_label_fn)
    df_train = pd.read_csv(wadi_data_abs, low_memory=False)
    df_test = pd.read_csv(wadi_label_abs, low_memory=False)

    # Process training data
    df_train.drop(columns=['Row', df_train.columns[-1]], inplace=True)
    df_train['Date'] = pd.to_numeric(pd.to_datetime(df_train['Date'], format='mixed'),downcast='float')
    df_train['Time'] = pd.to_numeric(pd.to_datetime(df_train['Time'], format='%M:%S.0'),downcast='float')
    df_train = df_train.astype(float)
    labels_train = []
    # Create directories
    make_dir_tree(os.path.join(config.wadi_data_dir, "Train", "Tracks"))
    make_dir_tree(os.path.join(config.wadi_data_dir, "Train", "Labels"))
    # Save data
    track_out_dir = os.path.join(config.wadi_data_dir, 'Train', 'Tracks')
    track_out_file = os.path.join(track_out_dir, '0.df.pkl')
    with open(track_out_file, 'wb') as file:
        pickle.dump(df_train, file)
    label_out_dir = os.path.join(config.wadi_data_dir, 'Train', 'Labels')
    label_out_file = os.path.join(label_out_dir, '0.l.pkl')
    with open(label_out_file, 'wb') as file:
        pickle.dump(labels_train, file)

    # Process testing data
    df_test.columns = [name.strip() for name in df_test.iloc[0,:]]
    df_test.drop(index=[0, 172803, 172802], inplace=True)
    labels_test = np.asarray(df_test.iloc[:,-1].astype(float))
    labels_test = np.abs(-1.*((labels_test - 1.)/2.))
    labels_test = labels_test.reshape(-1,1)
    labels_test = helpers.convert_anoms_from_tranad(labels_test)
    df_test.drop(columns=['Row', 'TOTAL_CONS_REQUIRED_FLOW', "Attack LABLE (1:No Attack, -1:Attack)"], inplace=True)
    df_test['Date'] = pd.to_numeric(pd.to_datetime(df_test['Date'], format='%m/%d/%y'),downcast='float')
    df_test['Time'] = pd.to_numeric(pd.to_datetime(df_test['Time'], format='%M:%S.0'),downcast='float')
    df_test = df_test.astype(float)
    # Create directories
    make_dir_tree(os.path.join(config.wadi_data_dir, "Test", "Tracks"))
    make_dir_tree(os.path.join(config.wadi_data_dir, "Test", "Labels"))
    # Save data
    track_out_dir = os.path.join(config.wadi_data_dir, 'Test', 'Tracks')
    track_out_file = os.path.join(track_out_dir, '0.df.pkl')
    with open(track_out_file, 'wb') as file:
        pickle.dump(df_test, file)
    label_out_dir = os.path.join(config.wadi_data_dir, 'Test', 'Labels')
    label_out_file = os.path.join(label_out_dir, '0.l.pkl')
    with open(label_out_file, 'wb') as file:
        pickle.dump(labels_test, file)
    
    # Combine train and test data
    combine_train_test_data('WADI')

    # Create new labels
    labels = shift_anoms(labels_test, df_train.shape[0])
    label_out_dir = os.path.join(config.wadi_data_dir, 'Both', 'Labels')
    label_out_file = os.path.join(label_out_dir, '0.l.pkl')
    # Create directory for labels
    make_dir_tree(os.path.join(config.wadi_data_dir, "Both", "Labels"))
    # Save labels
    label_out_dir = os.path.join(config.wadi_data_dir, 'Both', 'Labels')
    label_out_file = os.path.join(label_out_dir, '0.l.pkl')
    with open(label_out_file, 'wb') as file:
        pickle.dump(labels, file)

    return None

def initialize_smd_data() -> None:
    '''
    Initialize SMD data

    Returns: `None`

    NOT UNIT TESTED  
    Reason: This functino initializes the SMD dataset for use in the rest of the code. Unit testing this would mean doing the actual preprocessing and what not, which is better served as something to be done outside of testing.
    '''
    # Load data
    file_names = helpers.get_SMD_machine_names()

    # Column names
    col_names = [f"Column {i}" for i in range(38)]

    # Track shapes, maps file name to track shape
    track_shape_dict = dict()

    # Save train/test data
    for file_name in file_names:
        for split in ['Train', 'Test']:
            full_path = os.path.join(config.raw_data_dir, 'SMD', split.lower(), file_name)
            df = pd.read_csv(full_path, names=col_names, low_memory=False)
            track_out_dir = os.path.join(config.smd_data_dir, split, "Tracks")
            track_out_file = os.path.join(track_out_dir, f'{file_name.split(".")[0]}.df.pkl')
            make_dir_tree(track_out_dir)
            track_shape_dict[file_name] = df.shape
            with open(track_out_file, 'wb') as file:
                pickle.dump(df, file)

    # Combine train and test data
    combine_train_test_data('SMD')

    # Save label data
    for file_name in file_names:
        for split in ['Train', 'Test', 'Both']:
            if split in ['Test', 'Both']:
                # Load test label
                full_path = os.path.join(config.raw_data_dir, 'SMD', 'test_label', file_name)
                df = pd.read_csv(full_path, names=["Column 0"], low_memory=False)
                arr = np.asarray(df)
                labels_l = helpers.convert_anoms_from_tranad(arr)
                if split == 'Both':
                    labels_l = shift_anoms(labels_l, track_shape_dict[file_name][0])
            elif split == 'Train':
                labels_l = []
            labels_out_dir = os.path.join(config.smd_data_dir, split, "Labels")
            labels_out_file = os.path.join(labels_out_dir, f'{file_name.split(".")[0]}.l.pkl')
            make_dir_tree(labels_out_dir)
            with open(labels_out_file, 'wb') as file:
                pickle.dump(labels_l, file)

    return None

def initialize_swat_data() -> None:
    """
    Initialize SWaT data

    Returns: `None`

    NOT UNIT TESTED  
    Reason: This functino initializes the SWaT dataset for use in the rest of the code. Unit testing this would mean doing the actual preprocessing and what not, which is better served as something to be done outside of testing.
    """
    # Load data
    normal_path = os.path.join(config.raw_data_dir, 'SWaT', config.swat_normal_fn)
    normal_df = pd.read_csv(normal_path, low_memory=False)
    labels = []
    # Remove labels and timestamp
    normal_df.drop(["Timestamp", "Normal/Attack"], axis=1, inplace=True)
    # Convert to float
    normal_df = normal_df.map(lambda x: str(x).replace(",", ".")).astype(float)
    # Create directories
    make_dir_tree(os.path.join(config.swat_data_dir, "Train", "Tracks"))
    make_dir_tree(os.path.join(config.swat_data_dir, "Train", "Labels"))
    # Save data
    track_out_dir = os.path.join(config.swat_data_dir, 'Train', 'Tracks')
    track_out_file = os.path.join(track_out_dir, '0.df.pkl')
    with open(track_out_file, 'wb') as file:
        pickle.dump(normal_df, file)
    label_out_dir = os.path.join(config.swat_data_dir, 'Train', 'Labels')
    label_out_file = os.path.join(label_out_dir, '0.l.pkl')
    with open(label_out_file, 'wb') as file:
        pickle.dump(labels, file)

    # Load data
    attack_path = os.path.join(config.raw_data_dir, 'SWaT', config.swat_attack_fn)
    attack_df = pd.read_csv(attack_path, sep=";", low_memory=False)
    labels = [ float(label != 'Normal' ) for label in attack_df["Normal/Attack"].values]
    # Convert labels to start and end times
    labels = np.asarray([labels]).reshape(-1,1)
    labels = convert_anoms_from_tranad(labels)
    # Remove labels and timestamp from dataframe
    attack_df.drop(["Timestamp", "Normal/Attack"], axis=1, inplace=True)
    # Convert to float
    attack_df = attack_df.map(lambda x: str(x).replace(",", ".")).astype(float)
    # Create directories
    make_dir_tree(os.path.join(config.swat_data_dir, "Test", "Tracks"))
    make_dir_tree(os.path.join(config.swat_data_dir, "Test", "Labels"))
    # Save data
    track_out_dir = os.path.join(config.swat_data_dir, 'Test', 'Tracks')
    track_out_file = os.path.join(track_out_dir, '0.df.pkl')
    with open(track_out_file, 'wb') as file:
        pickle.dump(attack_df, file)
    label_out_dir = os.path.join(config.swat_data_dir, 'Test', 'Labels')
    label_out_file = os.path.join(label_out_dir, '0.l.pkl')
    with open(label_out_file, 'wb') as file:
        pickle.dump(labels, file)
    
    # Combine train and test data
    combine_train_test_data('SWaT')

    # Create new labels
    labels = shift_anoms(labels, normal_df.shape[0])
    label_out_dir = os.path.join(config.swat_data_dir, 'Both', 'Labels')
    label_out_file = os.path.join(label_out_dir, '0.l.pkl')
    # Create directory for labels
    make_dir_tree(os.path.join(config.swat_data_dir, "Both", "Labels"))
    # Save labels
    label_out_dir = os.path.join(config.swat_data_dir, 'Both', 'Labels')
    label_out_file = os.path.join(label_out_dir, '0.l.pkl')
    with open(label_out_file, 'wb') as file:
        pickle.dump(labels, file)

    return None

def initialize_smap_msl_data_all() -> None:
    """
    Initialize all training, testing, and combined data and create labels for SMAP/MSL

    Returns: `None`

    Tested (`initialize_smap_msl_data_all`)
    """
    channels = get_all_channels()
    initialize_smap_msl_data_channels(channels)
    return None

def initialize_smap_msl_data_channels(channels: list[str]) -> None:
    """
    Initialize provided training, testing, and combined data and create labels for SMAP/MSL
    Sample input: ['C-2', 'P-1']

    `channels`: List of strings containing channels for SMAP and MSL to initialize (list(str))  

    Returns: `None`

    Tested (`test_preprocess_data.test_initialize_smap_msl_data_channels`)
    """
    # Copy files
    for chan_id in channels:
        sc = get_sc_from_chan_id(chan_id)
        for split in ['Train', 'Test']:
            track_np = np.load(os.path.join(config.raw_data_dir, sc, split.lower(), f'{chan_id}.npy'))
            out_dir = os.path.join(config.val_dict[f'{sc.lower()}_data_dir'], split, 'Tracks')
            out_file = os.path.join(out_dir, f'{chan_id}.df.pkl')
            make_dir_tree(out_dir)
            with open(out_file, 'wb') as file:
                pickle.dump(pd.DataFrame(track_np), file)

    # Combine files
    combine_train_test_data('SMAP')
    combine_train_test_data('MSL')

    # Create labels
    for sc in ['SMAP', 'MSL']:
        for split in ['Test', 'Train', 'Both']:
            files = os.listdir(os.path.join(config.val_dict[f'{sc.lower()}_data_dir'], split, 'Tracks'))
            channels = [fn.split('.')[0] for fn in files]
            for chan_id in channels:
                label_df = pd.read_csv(os.path.join(config.raw_data_dir, sc, f'labeled_anomalies.csv'))
                label_indices = label_df[label_df['chan_id'] == chan_id]['anomaly_sequences'].iat[0]
                if split == 'Train':
                    labels = []
                elif split == 'Test':
                    labels = convert_str_to_int_list(label_indices)
                elif split == 'Both':
                    train_np = np.load(os.path.join(config.raw_data_dir, sc, 'train', f'{chan_id}.npy'))
                    train_shape = train_np.shape[0]
                    labels = shift_anoms(convert_str_to_int_list(label_indices), train_shape)

                out_dir = os.path.join(config.val_dict[f'{sc.lower()}_data_dir'], split, 'Labels')
                out_file = os.path.join(out_dir, f'{chan_id}.l.pkl')
                make_dir_tree(out_dir)
                with open(out_file, 'wb') as file:
                    pickle.dump(labels, file)

    return None

def initialize_dsn_data(size: Literal['DSN_100', 'DSN_1k']) -> None:
    """
    Split the given DSN data into separate drs and tracks files
    Tracks labeled as assigned integer
    DRS are labeled as the same integer as the track name
    If a DR already exists, increment counter by 1 (ie. 213_0.df.pkl -> 213_1.df.pkl)

    `size`: Which DSN dataset to use (1k or 100) (str)  

    Returns: `None`

    NOT UNIT TESTED  
    Reason: The split names are hard-coded with their actual file names in `helpers.get_dsn_split_names` and the test config class doesn't overwrite that, so I just test the function `initialize_dsn_data_split_names` instead.
    """
    if size not in ['DSN_100', 'DSN_1k']:
        raise Exception(f"DSN data {size} not valid, please use either \'DSN_100\' or \'DSN_1k\'")

    # Go through and copy files to 'Train' and 'Test' depending on split
    train_tracks, test_tracks = helpers.get_dsn_split_names(size)

    initialize_dsn_data_split_names(size, train_tracks, test_tracks)

    return None

def initialize_dsn_data_split_names(size: Literal['DSN_100', 'DSN_1k'], train_tracks, test_tracks) -> None:
    """
    Split the given DSN data into separate drs and tracks files
    Tracks labeled as assigned integer
    DRS are labeled as the same integer as the track name
    If a DR already exists, increment counter by 1 (ie. 213_0.df.pkl -> 213_1.df.pkl)

    Training and testing file names are provided

    `size`: Which DSN dataset to use (1k or 100) (str)  
    `train_tracks`: List of DSN training tracks (str)  
    `test_tracks`: List of DSN testing tracks (str)  

    Returns: `None`

    Tested (`test_preprocess_data.initialize_dsn_data_split_names`)
    """

    if size not in ['DSN_100', 'DSN_1k']:
        raise Exception(f"DSN data {size} not valid, please use either \'DSN_100\' or \'DSN_1k\'")

    # Create output directories
    for split in ['Train', 'Test', 'Both']:
        for type in ['Labels', 'Tracks']:
            make_dir_tree(os.path.join(config.val_dict[f'{size.lower()}_data_dir'], split, type))

    # Load raw DSN data
    mons, drs = load_raw_dsn_data(size)

    # Save each track as its ID as a pickled DF to 'Both'
    for i, track in enumerate(mons):
        if track[0]==17894088 and size=='DSN_1k':
            print("initialize_dsn_data (DSN_1k): Skipping track 17894088 since it has 103 parameters but all other tracks have 129 parameters.")
            continue
        track_out = track[1]
        fname = os.path.join(config.val_dict[f'{size.lower()}_data_dir'], 'Both', 'Tracks', f'{track[0]}.df.pkl')
        with open(fname, 'wb') as file:
            pickle.dump(track_out, file)

    # Collect labels for each track and save it to the corresponding track ID (save to 'Both')
    drs_dict = gen_drs_dict(mons, drs)
    for mons_idx in range(len(mons)):
        labels = list()
        track_df = mons[mons_idx][1]
        if mons_idx in drs_dict.keys():
            for dr_idx in drs_dict[mons_idx]:
                dr_df = drs[dr_idx][1]
                anomaly_start_time = dr_df.at[0, 'INCIDENT_START_TIME_DT']
                anomaly_end_time = dr_df.at[0, 'INCIDENT_END_TIME_DT']
                start_idx = 0
                end_idx = 0
                max_valid_idx = track_df.shape[0]-1
                for j in track_df.index:
                    if track_df.iat[j,0] <= anomaly_start_time:
                        start_idx = j
                    if track_df.iat[j,0] <= anomaly_end_time:
                        end_idx = j
                # Have end_idx cover entire anomaly range, even in one step isn't 100% an anomaly
                if end_idx < max_valid_idx and\
                    (track_df.iat[end_idx,0] < anomaly_end_time < track_df.iat[end_idx+1,0]):
                    end_idx += 1

                # Add label
                labels.append([start_idx, end_idx])

        # Save labels
        fname = os.path.join(config.val_dict[f'{size.lower()}_data_dir'], 'Both', 'Labels', f'{mons[mons_idx][0]}.l.pkl')
        with open(fname, 'wb') as file:
            pickle.dump(labels, file)

    for train_track in train_tracks:
        # Copy track
        src_file = os.path.join(config.val_dict[f'{size.lower()}_data_dir'], 'Both', 'Tracks', f'{train_track}.df.pkl')
        dst_file = os.path.join(config.val_dict[f'{size.lower()}_data_dir'], 'Train', 'Tracks', f'{train_track}.df.pkl')
        shutil.copy(src_file, dst_file)

        # Copy label
        src_file = os.path.join(config.val_dict[f'{size.lower()}_data_dir'], 'Both', 'Labels', f'{train_track}.l.pkl')
        dst_file = os.path.join(config.val_dict[f'{size.lower()}_data_dir'], 'Train', 'Labels', f'{train_track}.l.pkl')
        shutil.copy(src_file, dst_file)

    for test_track in test_tracks:
        # Copy track
        src_file = os.path.join(config.val_dict[f'{size.lower()}_data_dir'], 'Both', 'Tracks', f'{test_track}.df.pkl')
        dst_file = os.path.join(config.val_dict[f'{size.lower()}_data_dir'], 'Test', 'Tracks', f'{test_track}.df.pkl')
        shutil.copy(src_file, dst_file)

        # Copy label
        src_file = os.path.join(config.val_dict[f'{size.lower()}_data_dir'], 'Both', 'Labels', f'{test_track}.l.pkl')
        dst_file = os.path.join(config.val_dict[f'{size.lower()}_data_dir'], 'Test', 'Labels', f'{test_track}.l.pkl')
        shutil.copy(src_file, dst_file)

    return None

def preprocess_numpy(track_df: pd.DataFrame,\
                     byte_vals: list[list[Any]] = None,\
                     byte_mod: Literal['one-hot', 'byte-dict', 'constant'] = 'constant',\
                     nan_value: float = 0.0)\
                        -> tuple[np.ndarray, Any]:
    """
    Convert track to (time x parameter) 2D numpy array of floating point values
    Changes byte values to the ones generated in byte_dict or one hot encoding

    Outputs tuple, first element is the track

    `track_df`: A track's dataframe to convert for use in plotly (pd.DataFrame)  
    `byte_vals`: List of unique byte values encountered per-parameter (list(list(byte value)))  
    `byte_mod`: How to deal with byte values (str)  
    `nan_value`: What to replace NAN values with (float)

    Returns: `tuple[np.ndarray, Any]`  
    
    Tested (`test.test_preprocess_numpy`)
    """
    # Convert byte values appropriately
    if byte_mod == 'constant':
        track_df = replace_byte_constant(track_df)
    elif byte_mod == 'byte-dict':
        track_df = replace_byte_dict(track_df, byte_vals)
    elif byte_mod == 'one-hot':
        track_df = replace_byte_one_hot(track_df, byte_vals)
    else:
        raise Exception(f"byte_mod value {byte_mod} invalid, use \'constant\', \'byte-dict\', or \'one-hot\'.")
    # Convert all nans to predefined value
    track_df = track_df.fillna(float(nan_value))
    # Get underlying numpy array from pandas as np.float64
    track_np = track_df.to_numpy(dtype=np.float64)
    return track_np

def preprocess_plotly(track: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all bytes in DataFrame to strings, returning a DataFrame as well

    `track`: A track's dataframe to convert for use in plotly  

    Returns: `pd.DataFrame`
    
    Tested (`test.test_preprocess_plotly`)
    """
    # Make copy of the track
    track_out = track
    # Convert bytes to strings
    columns = track_out.columns
    byte = type(b'X')
    for column in columns:
        if type(track_out[column][0]) == byte:
            str_col = track_out[column].str.decode("utf-8")
            track_out[column] = str_col

    return track_out

def preprocess_data(dataset: Literal['DSN_100', 'DSN_1k', 'SMAP', 'MSL', 'SWaT', 'WADI', 'SMD'],\
                    byte_mod: Literal['constant','byte-dict','one-hot'] = 'constant')\
    -> None:
    """
    Converts a dataset to the specified format below.

    track_i is a (time x parameter) 2D numpy array of floating point values
    and labels_i is a list containing anomaly start and end times: [[anom1_s, anom1_e], [anom2_s, anom2_e], ...]

    `dataset`: Which dataset to preprocess (str)  
    `byte_mod`: How to deal with byte values (str)  

    Returns: `None`

    Tested (test.test_preprocess_data)
    """
    # Check inputs
    if dataset not in ['DSN_100', 'DSN_1k', 'SMAP', 'MSL', 'SWaT', 'WADI', 'SMD']:
        raise Exception(f"Error: Invalid dataset option \"{dataset}\"")
    if byte_mod not in ['constant', 'byte-dict', 'one-hot']:
        raise Exception(f"byte_mod value {byte_mod} invalid, use \'constant\', \'byte-dict\', or \'one-hot\'.")

    # Set splits based on datasets
    if dataset in ['SMAP', 'MSL', 'SWaT', 'SMD', 'WADI', 'DSN_100', 'DSN_1k']:
        splits = ['Train', 'Test', 'Both']

    # Prepare output directories
    for split in splits:
        preprocessed_dir = os.path.join(config.preprocess_save_dir, dataset, split)
        preprocessed_labels = os.path.join(preprocessed_dir, 'Labels')
        preprocessed_tracks = os.path.join(preprocessed_dir, 'Tracks')
        # Make directory tree
        make_dir_tree([preprocessed_labels, preprocessed_tracks])

    # Iterate over splits
    for split in splits:
        # Iterate over tracks
        names = get_dataset_split_track_names(dataset, split=split)
        for i in tqdm(range(len(names))):
            track_name = names[i]

            track_path = os.path.join(config.val_dict[f'{dataset.lower()}_data_dir'], split, 'Tracks', f'{track_name}.df.pkl')
            with open(track_path, 'rb') as file:
                track_df = pickle.load(file)

            label_path = os.path.join(config.val_dict[f'{dataset.lower()}_data_dir'], split, 'Labels', f'{track_name}.l.pkl')
            with open(label_path, 'rb') as file:
                labels = pickle.load(file)

            byte_val_path = os.path.join(config.parse_save_dir, dataset, 'Both', 'pkl', f'byte_vals.pkl')
            with open(byte_val_path, 'rb') as file:
                byte_vals = pickle.load(file)

            track_out = preprocess_numpy(track_df, byte_vals=byte_vals, byte_mod=byte_mod)
            labels = helpers.convert_anoms_to_tranad(labels, track_out.shape)

            # Write to output
            track_output_dir = os.path.join(config.preprocess_save_dir, dataset, split, 'Tracks')
            labels_output_dir = os.path.join(config.preprocess_save_dir, dataset, split, 'Labels')
            np.save(os.path.join(track_output_dir, f"{track_name}.npy"), track_out)
            np.save(os.path.join(labels_output_dir, f"{track_name}.npy"), labels)

    return None

