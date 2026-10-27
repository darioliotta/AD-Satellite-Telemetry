###############################
# Imports # Imports # Imports #
###############################

import os
from pathlib import Path

###################################
# Initialization # Initialization #
###################################

def init():
    """
    Function called once to initialize the Config class

    Note: SERIOUSLY! Call this only once every time you run something. Every time it's initialized it gets re-written from scratch.

    Tested (test_global_config.test_config_class)
    """
    global config
    config = Config()

###############################
# Config Class # Config Class #
###############################

class Config:
    """
    Config class helps manage all the different directories and hyperparameters used in this repo.

    Initialize once to instantiate the global variable, then you can call the init_test() function if you want to load test-specific configuration values.

    Tested (test_global_config.test_config_class)
    """
    def __init__(self) -> None:
        # Locate parent directory
        self.top_dir = str(Path(__file__).parent.parent.absolute())

        # Package name
        self.package_name = "TranADPlus"

        # Value dictionary used for easy indexing
        self.val_dict = dict()

        # Initialize normal values by defualt
        self.init_normal()

        return None

    def update_val_dict(self) -> None:
        """
        Automatically called after `init_testing()` and `init_normal()`.
        Call this to update `self.val_dict` with current class variable values.

        Returns: `None`

        Tested (test_global_config.test_config_class)
        """
        ######################
        # MISC # MISC # MISC #
        ######################

        self.val_dict["package_name"] = self.package_name
        self.val_dict["top_dir"] = self.top_dir

        ###################################################
        # Dataset File Structure # Dataset File Structure # 
        ###################################################

        # Directory structure
        self.val_dict["dataset_dir"] = self.dataset_dir
        self.val_dict["checkpoint_dir"] = self.checkpoint_dir
        self.val_dict["pickle_dir"] = self.pickle_dir
        self.val_dict["raw_data_dir"] = self.raw_data_dir
        self.val_dict["original_data_dir"] = self.original_data_dir
        self.val_dict["preprocess_save_dir"] = self.preprocess_save_dir
        self.val_dict["parse_save_dir"] = self.parse_save_dir

        # SMAP/MSL dataset parameters
        self.val_dict["smap_data_dir"] = self.smap_data_dir
        self.val_dict["msl_data_dir"] = self.msl_data_dir

        # DSN_100 dataset parameters
        self.val_dict["dsn_100_data_dir"] = self.dsn_100_data_dir
        self.val_dict["mons_100_pkl_fn"] = self.mons_100_pkl_fn
        self.val_dict["drs_100_pkl_fn"] = self.drs_100_pkl_fn

        # DSN 1k dataset parameters
        self.val_dict["dsn_1k_data_dir"] = self.dsn_1k_data_dir
        self.val_dict["mons_1k_pkl_fn"] = self.mons_1k_pkl_fn
        self.val_dict["drs_1k_pkl_fn"] = self.drs_1k_pkl_fn

        # SWaT dataset parameters
        self.val_dict["swat_data_dir"] = self.swat_data_dir
        self.val_dict["swat_attack_fn"] = self.swat_attack_fn
        self.val_dict["swat_normal_fn"] = self.swat_normal_fn

        # WADI dataset parameters
        self.val_dict["wadi_data_dir"] = self.wadi_data_dir
        self.val_dict["wadi_data_fn"] = self.wadi_data_fn
        self.val_dict["wadi_label_fn"] = self.wadi_label_fn

        # SMD dataset parameters
        self.val_dict["smd_data_dir"] = self.smd_data_dir

        ###################################################
        # Submodules Directories # Submodules Directories #
        ###################################################

        # Submodules - optional (I used this for exporting preprocessed datasets to those directories)
        #self.val_dict["TranAD_dir"] = self.TranAD_dir
        #self.val_dict["OmniAnomaly_dir"] = self.OmniAnomaly_dir
        #self.val_dict["MTAD_GAT_dir"] = self.MTAD_GAT_dir

        #######################################################
        # Training Hyperparameters # Training Hyperparameters #
        #######################################################

        self.val_dict["USAD"] = self.USAD
        self.val_dict["TranAD"] = self.TranAD

        #################################
        # Test-Specific # Test-Specific #
        #################################

        self.val_dict['test_config_val'] = self.test_config_val

        return None

    def init_testing(self) -> None:
        """
        Initialize testing configuration values

        Returns: `None`

        Tested (test_global_config.test_config_class)
        """

        ###################################################
        # Dataset File Structure # Dataset File Structure # 
        ###################################################

        # Directory structure
        self.dataset_dir = os.path.join(self.top_dir, "test", "Datasets")
        self.checkpoint_dir = os.path.join(self.top_dir, "test", "Checkpoints")
        self.pickle_dir = os.path.join(self.top_dir, "test", "Pickles")
        self.raw_data_dir = os.path.join(self.dataset_dir, "Raw")
        self.original_data_dir = os.path.join(self.dataset_dir, "Original")
        self.preprocess_save_dir = os.path.join(self.dataset_dir, "Preprocessed")
        self.parse_save_dir = os.path.join(self.dataset_dir, "EDA")

        # SMAP/MSL dataset parameters
        self.smap_data_dir = os.path.join(self.original_data_dir, "SMAP")
        self.msl_data_dir = os.path.join(self.original_data_dir, "MSL")

        # DSN_100 dataset parameters
        self.dsn_100_data_dir = os.path.join(self.original_data_dir, "DSN_100")
        self.mons_100_pkl_fn = "mons_anonymized.pkl"
        self.drs_100_pkl_fn = "drs.pkl"

        # DSN 1k dataset parameters
        self.dsn_1k_data_dir = os.path.join(self.original_data_dir, "DSN_1k")
        self.mons_1k_pkl_fn = "mons.pkl"
        self.drs_1k_pkl_fn = "drs.pkl"

        # SWaT dataset parameters
        self.swat_data_dir = os.path.join(self.original_data_dir, "SWaT")
        self.swat_attack_fn = "SWaT_Dataset_Attack_v0.csv"
        self.swat_normal_fn = "SWaT_Dataset_Normal_v1.csv"

        # WADI dataset parameters
        self.wadi_data_dir = os.path.join(self.original_data_dir, "WADI")
        self.wadi_data_fn = "WADI.A2_19 Nov 2019/WADI_14days_new.csv"
        self.wadi_label_fn = "WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv"

        # SMD dataset parameters
        self.smd_data_dir = os.path.join(self.original_data_dir, "SMD")

        #################################
        # Test-Specific # Test-Specific #
        #################################

        # Values used in unit-tests
        self.test_config_val = 0xdeadbeef

        #####################################
        # Update Val Dict # Update Val Dict #
        #####################################

        # Update value dictionary
        self.update_val_dict()

        return None

    def init_normal(self) -> None:
        """
        Initialize normal configuration values

        Returns: `None`

        Tested (test_global_config.test_config_class)
        """

        ###################################################
        # Dataset File Structure # Dataset File Structure # 
        ###################################################

        # Directory structure
        self.dataset_dir = os.path.join(self.top_dir, "Datasets")
        self.checkpoint_dir = os.path.join(self.top_dir, "Checkpoints")
        self.pickle_dir = os.path.join(self.top_dir, "Pickles")
        self.raw_data_dir = os.path.join(self.dataset_dir, "Raw")
        self.original_data_dir = os.path.join(self.dataset_dir, "Original")
        self.preprocess_save_dir = os.path.join(self.dataset_dir, "Preprocessed")
        self.parse_save_dir = os.path.join(self.dataset_dir, "EDA")

        # SMAP/MSL dataset parameters
        self.smap_data_dir = os.path.join(self.original_data_dir, "SMAP")
        self.msl_data_dir = os.path.join(self.original_data_dir, "MSL")

        # DSN_100 dataset parameters
        self.dsn_100_data_dir = os.path.join(self.original_data_dir, "DSN_100")
        self.mons_100_pkl_fn = "mons_anonymized.pkl"
        self.drs_100_pkl_fn = "drs.pkl"

        # DSN 1k dataset parameters
        self.dsn_1k_data_dir = os.path.join(self.original_data_dir, "DSN_1k")
        self.mons_1k_pkl_fn = "mons.pkl"
        self.drs_1k_pkl_fn = "drs.pkl"

        # SWaT dataset parameters
        self.swat_data_dir = os.path.join(self.original_data_dir, "SWaT")
        self.swat_attack_fn = "SWaT_Dataset_Attack_v0.csv"
        self.swat_normal_fn = "SWaT_Dataset_Normal_v1.csv"

        # WADI dataset parameters
        self.wadi_data_dir = os.path.join(self.original_data_dir, "WADI")
        self.wadi_data_fn = "WADI.A2_19 Nov 2019/WADI_14days_new.csv"
        self.wadi_label_fn = "WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv"

        # SMD dataset parameters
        self.smd_data_dir = os.path.join(self.original_data_dir, "SMD")

        #################################
        # Test-Specific # Test-Specific #
        #################################

        # Values used in unit-tests
        self.test_config_val = 0x12345678

        ###################################################
        # Submodules Directories # Submodules Directories #
        ###################################################

        # Submodules - optional (I used this for exporting preprocessed datasets to those directories)
        #self.TranAD_dir = os.path.join(self.top_dir, "submodules", "TranAD")
        #self.OmniAnomaly_dir = os.path.join(self.top_dir, "submodules", "OmniAnomaly")
        #self.MTAD_GAT_dir = os.path.join(self.top_dir, "submodules", "mtad-gat-pytorch")

        #######################################################
        # Training Hyperparameters # Training Hyperparameters #
        #######################################################

        self.USAD = {
            "SWaT":{
                "lr": 0.001,
                "epochs": 70,
                "batch_sz": 8192,
                "window_sz": 12,
                "hidden_sz": 40,
                "pot_q": 0.001,
                "shuffle": False,
                "padding": False,
                "num_workers": 0,
                "downsample": 5
            },
            "WADI":{
                "lr": 0.001,
                "epochs": 70,
                "batch_sz": 8192,
                "window_sz": 10,
                "hidden_sz": 100,
                "pot_q": 0.001,
                "shuffle": False,
                "padding": False,
                "num_workers": 0,
                "downsample": 5
            },
            "SMD":{
                "lr": 0.001,
                "epochs": 250,
                "batch_sz": 8192,
                "window_sz": 5,
                "hidden_sz": 38,
                "pot_q": 0.001,
                "shuffle": False,
                "padding": False,
                "num_workers": 0,
                "downsample": 5
            },
            "SMAP":{
                "lr": 0.001,
                "epochs": 250,
                "batch_sz": 8192,
                "window_sz": 5,
                "hidden_sz": 55,
                "pot_q": 0.07,
                "shuffle": False,
                "padding": False,
                "num_workers": 0,
                "downsample": 5
            },
            "MSL":{
                "lr": 0.001,
                "epochs": 250,
                "batch_sz": 8192,
                "window_sz": 5,
                "hidden_sz": 33,
                "pot_q": 0.01,
                "shuffle": False,
                "padding": False,
                "num_workers": 0,
                "downsample": 5
            },
            "DSN_1k":{
                "lr": 0.001,
                "epochs": 250,
                "batch_sz": 8192,
                "window_sz": 5,
                "hidden_sz": 33,
                "pot_q": 0.001,
                "shuffle": False,
                "padding": False,
                "num_workers": 0,
                "downsample": 5
            }
        }

        self.TranAD = {
            "SWaT":{
                "lr": 0.008,
                "epochs": 5,
                "batch_sz": 8192,
                "window_sz": 10,
                "dim_feedforward": 16,
                "pot_q": 0.001,
                "shuffle": False,
                "padding": True,
                "num_workers": 0,
                "num_layers": 1,
                "downsample": None
            },
            "WADI":{
                "lr": 0.0001,
                "epochs": 5,
                "batch_sz": 8192,
                "window_sz": 10,
                "dim_feedforward": 16,
                "pot_q": 0.001,
                "shuffle": False,
                "padding": True,
                "num_workers": 0,
                "num_layers": 1,
                "downsample": None
            },
            "SMD":{
                "lr": 0.0001,
                "epochs": 5,
                "batch_sz": 8192,
                "window_sz": 10,
                "dim_feedforward": 16,
                "pot_q": 0.001,
                "shuffle": False,
                "padding": True,
                "num_workers": 0,
                "num_layers": 1,
                "downsample": None
            },
            "SMAP":{
                "lr": 0.001,
                "epochs": 5,
                "batch_sz": 8192,
                "window_sz": 10,
                "dim_feedforward": 16,
                "pot_q": 0.07,
                "shuffle": False,
                "padding": True,
                "num_workers": 0,
                "num_layers": 1,
                "downsample": None
            },
            "MSL":{
                "lr": 0.002,
                "epochs": 5,
                "batch_sz": 8192,
                "window_sz": 10,
                "dim_feedforward": 16,
                "pot_q": 0.01,
                "shuffle": False,
                "padding": True,
                "num_workers": 0,
                "num_layers": 1,
                "downsample": None
            },
            "DSN_1k":{
                "lr": 0.0001,
                "epochs": 5,
                "batch_sz": 8192,
                "window_sz": 10,
                "dim_feedforward": 16,
                "pot_q": 0.001,
                "shuffle": False,
                "padding": True,
                "num_workers": 0,
                "num_layers": 1,
                "downsample": None
            }
        }

        #####################################
        # Update Val Dict # Update Val Dict #
        #####################################

        # Update value dictionary
        self.update_val_dict()

        return None