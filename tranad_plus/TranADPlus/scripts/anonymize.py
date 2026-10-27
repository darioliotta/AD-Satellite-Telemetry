##########
# Config #
##########

import os
import sys
import pickle
import random
import string
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
from TranADPlus.src import helpers
import TranADPlus.src.global_config as global_config

# Load config
config = global_config.config

#################################
# Import DSN_1k in its entirety #
#################################

mons, drs = helpers.load_raw_dsn_data('DSN_1k')
new_mons = list()
new_drs = list()

########################
# Load Byte Dictionary #
########################

# Get list of unique byte values encountered
byte_val_path = os.path.join(config.val_dict[f'parse_save_dir'], 'DSN_1k', 'Both', 'pkl', f'byte_vals.pkl')
with open(byte_val_path, 'rb') as file:
    byte_vals = pickle.load(file)
byte_vals_small = []
for byte_vals_parameter in byte_vals:
    for parameter_byte_value in byte_vals_parameter:
        byte_vals_small.append(str(parameter_byte_value))
byte_vals_small = list(set(byte_vals_small))

#########################
# Randomize Byte Values #
#########################

# Create a random list of strings of length N
# Source:
# https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
random_byte_vals = set()
N = 4
while len(random_byte_vals) < len(byte_vals_small):
    new_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
    random_byte_vals = random_byte_vals.union([new_str])
random_byte_vals = list(random_byte_vals)

####################
# Get Column Names #
####################

col_names = mons[0][1].columns
col_names = list(col_names)

##########################
# Randomize Column Names #
##########################

# Create a random list of strings of length N
# Source:
# https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
random_col_names = set()
N = 4
while len(random_col_names) < len(col_names):
    new_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
    random_col_names = random_col_names.union([new_str])
random_col_names = list(random_col_names)

########################
# Iterate through mons #
########################

for cur_mons in mons:
    cur_mons_id = cur_mons[0]
    cur_mons_df = cur_mons[1].copy()

    if cur_mons_id == 17894088:
        # Has 103 parameters instead of 129
        continue

    ########################
    # Replace Column Names #
    ########################

    cur_mons_df.columns = random_col_names

    #######################
    # Replace byte values #
    #######################

    for column, dtype in enumerate(cur_mons_df.dtypes):
        if dtype == np.dtype('O'):
            map_fn = lambda byte_val: random_byte_vals[byte_vals_small.index(byte_val)]
            cur_mons_df.iloc[:,column] = cur_mons_df.iloc[:,column].map(map_fn)
        
    new_mons.append((cur_mons_id, cur_mons_df))

########################
# Get DRS Column Names #
########################

col_names = drs[0][1].columns
col_names = list(col_names)

########################
# Iterate through drs #
########################

for cur_dr in drs:
    cur_dr_id = cur_dr[0]
    cur_dr_df = cur_dr[1].copy()

    if cur_dr_id == 17894088:
        # Has 103 parameters instead of 129
        continue

    ####################
    # Drop DRs Columns #
    ####################

    cur_dr_df = cur_dr_df.iloc[:,1:5]

    new_drs.append((cur_dr_id, cur_dr_df))

# Save output
new_mons_path = 'new_mons.pkl'
with open(new_mons_path, 'wb') as f:
    pickle.dump(new_mons, f)

new_drs_path = 'new_drs.pkl'
with open(new_drs_path, 'wb') as f:
    pickle.dump(new_drs, f)

print("done")
