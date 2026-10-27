# Helper script to remove unfinished runs from specified directories (`ckpt_dir`, `pkl_dir`, `log_dir`)
# Remove all files associated with a file-name if:
# - Checkpoint doesn't have 'hp_dict'
# - Associated pickle files don't exist

import os
import sys
import torch
import pprint
import pickle
import shutil
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Update path to include TranADPlus
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

from TranADPlus.src import *

# Load config file
config = global_config.config

# Resulting directories
ckpt_dir = os.path.abspath(os.path.join(config.top_dir, '..', '..', 'Results', 'TranAD', 'Experiment1', 'Checkpoints'))
pkl_dir = os.path.abspath(os.path.join(config.top_dir, '..', '..', 'Results', 'TranAD', 'Experiment1', 'Pickles'))
log_dir = os.path.abspath(os.path.join(config.top_dir, '..', '..', 'Results', 'TranAD', 'Experiment1', 'Logs'))

# List all checkpoint files
files = os.listdir(ckpt_dir)

# Go through each one and delete unfinished runs
for fname in files:
    # Get filename without extension
    fname = fname.split('.')[0]
    print(fname)

    # Load checkpoint
    ckpt_path = os.path.join(ckpt_dir, fname + '.ckpt')
    ckpt = torch.load(ckpt_path)

    try:
        # Check if checkpoint is completed
        ds = ckpt['hp_dict']['dataset_str']

        # Check pickle files
        for method in ['bf', 'pot']:
            pkl_path = os.path.join(pkl_dir, fname + f'_{method}.pkl')
            if not os.path.isfile(pkl_path):
                raise Exception(f"Pickle file {pkl_path} doesn't exist")
    except Exception as e:
        print("File", fname, "wasn't complete, removing...")

        log_path_1 = os.path.join(log_dir, fname + '_train.txt')
        log_path_2 = os.path.join(log_dir, fname + '_eval.txt')

        pkl_path_1 = os.path.join(pkl_dir, fname + '_bf.pkl')
        pkl_path_2 = os.path.join(pkl_dir, fname + '_pot.pkl')

        if os.path.isfile(ckpt_path):
            os.remove(ckpt_path)

        if os.path.isfile(log_path_1):
            os.remove(log_path_1)

        if os.path.isfile(log_path_2):
            os.remove(log_path_2)

        if os.path.isfile(pkl_path_1):
            os.remove(pkl_path_1)

        if os.path.isfile(pkl_path_2):
            os.remove(pkl_path_2)
