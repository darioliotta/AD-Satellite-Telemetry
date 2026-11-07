"""TOAST v12 supported datasets and *unchanged* per-dataset v9 baselines.

Experiment overrides are frozen in experiment_plan_v12.json. This is not a
repository containing multiple mutually incompatible active campaigns.
"""
from pathlib import Path
import os
from project_paths import (CODE_ROOT, PROJECT_ROOT, DATA_ROOT, RAW_ROOT, PREPROCESSED_ROOT,
                           RESULTS_ROOT, CAMPAIGN_LOG_ROOT, dataset_result, checkpoint_dir,
                           pickle_dir, native_environment)

BASE_DIR = CODE_ROOT
PROJECT_DIR = PROJECT_ROOT
NATIVE_PACKAGE = CODE_ROOT / 'TranADPlus'
DATA_DIR = DATA_ROOT
RAW_DATA_DIR = RAW_ROOT
PROCESSED_DATA_DIR = PREPROCESSED_ROOT
RESULTS_DIR = RESULTS_ROOT
LOGS_DIR = CAMPAIGN_LOG_ROOT
DATASETS = ('OPS-SAT', 'SMAP', 'MSL')

COMMON = {'num_layers': 1, 'shuffle': False, 'padding': True,
          'num_workers': 0, 'downsample': None}

PROFILES = {
    'OPS-SAT': {
        'opssat_shared_ws10_1f': {
            'epochs': 50,'batch_sz': 256,'window_sz': 10,
            'lr': 0.001,'dim_feedforward': 8,'pot_q': 0.01,
            'seed': 2026, 'segments_csv': str(DATA_DIR / 'raw' / 'OPSSAT' / 'segments.csv'),
            'channels': None, 'max_train_fragments_per_channel': None,
            'max_test_fragments_per_channel': None}},
    'SMAP': {
        'smap_plus_ws10': {
            'epochs': 50,'batch_sz': 1024,'window_sz': 10,
            'lr': 0.001,'dim_feedforward': '2x','pot_q': 0.07}},
    'MSL': {
        'msl_plus_ws50_dff2x': {
            'epochs': 50,'batch_sz': 128,'window_sz': 50,
            'lr': 0.0001,'dim_feedforward': '2x','pot_q': 0.01}}
}
if os.environ.get('TRANADPLUS_CAMPAIGN_PLAN'):
    from campaign_plan import register_plan_profiles
    register_plan_profiles(PROFILES,os.environ['TRANADPLUS_CAMPAIGN_PLAN'])
EXPECTED_TRACK_COUNTS = {'SMAP': 54, 'MSL':28}


def list_profiles(dataset=None):
    selected=(dataset,) if dataset else DATASETS
    for name in selected:
        if name not in PROFILES:raise ValueError(f'Unsupported dataset: {name}')
        for identifier in PROFILES[name]:yield name,identifier


def get_profile(identifier):
    for dataset,profiles in PROFILES.items():
        if identifier in profiles:
            return {**COMMON,**profiles[identifier], 'dataset':dataset,'identifier':identifier}
    raise ValueError(f'Unknown training profile: {identifier}')


def checkpoint_path(identifier):
    return checkpoint_dir(get_profile(identifier)['dataset']) / f'{identifier}.ckpt'


def result_directory(dataset):
    if dataset not in DATASETS:raise ValueError(f'Unsupported dataset: {dataset}')
    return dataset_result(dataset)


def log_directory(dataset):
    if dataset not in DATASETS:raise ValueError(f'Unsupported dataset: {dataset}')
    return LOGS_DIR / dataset
