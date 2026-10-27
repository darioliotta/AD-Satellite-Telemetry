#####################
# Imports # Imports #
#####################

import os
from pathlib import Path
import sys
import torch
import time
import numpy as np
import argparse
from torch.utils.data import DataLoader

# Update path to include TranADPlus
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

from TranADPlus.src import global_config, models, helpers, print_data
from TranADPlus.src.datasets import TSDataset_tracks

# Torch set default data type to double precision (Causes error in optimizer.step())
#torch.set_default_dtype(torch.float64)

###################################################
# Main Training Function # Main Training Function # 
###################################################

def main(args=None):
    # Load config
    config = global_config.config
    top_dir = config.top_dir

    # Load data onto GPU if available
    # Note: device is switched from command line argument too
    if not torch.cuda.is_available():
        print(f"Warning: CUDA unavailable. Using CPU-only training instead.")
        device = 'cpu'
    else:
        device = 'cuda:0'

    # Create location where we store checkpoints
    checkpoint_dir = os.path.join(top_dir, 'Checkpoints')
    helpers.make_dir_tree(checkpoint_dir)

    ###############################
    # Asserts # Asserts # Asserts #
    ###############################

    if(args.dataset_str is None):
        raise Exception("training_loop.py: dataset_str is None")
    if(args.model_str is None):
        raise Exception("training_loop.py: model_str is None")
    if(args.identifier is None):
        raise Exception("training_loop.py: identifier is None")
    if(args.scaling is None):
        raise Exception("training_loop.py: scaling is None")

    dataset_str = args.dataset_str
    model_str = args.model_str

    #####################################
    # Hyperparameters # Hyperparameters #
    #####################################

    # Generic hyperparameters from config
    hp_dict = dict() # Store hyperparameters to print them easy
    hp_dict['dataset_str'] = dataset_str
    hp_dict['model_str'] = model_str
    hp_dict['identifier'] = args.identifier
    hp_dict['scaling'] = args.scaling
    hp_dict['features'] = helpers.get_n_features_from_dataset(args.dataset_str)
    hp_dict['lr'] = config.val_dict[model_str][dataset_str]['lr']
    hp_dict['epochs'] = config.val_dict[model_str][dataset_str]['epochs']
    hp_dict['batch_sz'] = config.val_dict[model_str][dataset_str]['batch_sz'] # larger batch size -> increase learning rate
    hp_dict['window_sz'] = config.val_dict[model_str][dataset_str]['window_sz']
    hp_dict['shuffle'] = config.val_dict[model_str][dataset_str]['shuffle']
    hp_dict['padding'] = config.val_dict[model_str][dataset_str]['padding']
    hp_dict['num_workers'] = config.val_dict[model_str][dataset_str]['num_workers']
    ds = config.val_dict[model_str][dataset_str]['downsample']
    hp_dict['downsample'] = int(ds) if ds != None else None
    hp_dict['pot_q'] = config.val_dict[model_str][dataset_str]['pot_q']
    if[model_str] == 'TranAD':
        hp_dict['dim_feedforward'] = config[model_str][dataset_str]['dim_feedforward']
        hp_dict['num_layers'] = config[model_str][dataset_str]['num_layers']
    if[model_str] == 'USAD':
        hp_dict['hidden_sz'] = config[model_str][dataset_str]['hidden_sz']

    # Overwrite from CLI
    if args is not None:
        if args.lr is not None:
            hp_dict['lr'] = args.lr
        if args.epochs is not None:
            hp_dict['epochs'] = args.epochs
        if args.batch_sz is not None:
            hp_dict['batch_sz'] = args.batch_sz
        if args.window_sz is not None:
            hp_dict['window_sz'] = args.window_sz
        if args.hidden_sz is not None:
            hp_dict['hidden_sz'] = args.hidden_sz
        if args.dim_feedforward is not None:
            hp_dict['dim_feedforward'] = args.dim_feedforward
        if args.num_layers is not None:
            hp_dict['num_layers'] = args.num_layers
        if args.pot_q is not None:
            hp_dict['pot_q'] = args.pot_q
        if args.shuffle is not None:
            hp_dict['shuffle'] = True if args.shuffle == 1 else False
        if args.padding is not None:
            hp_dict['padding'] = True if args.padding == 1 else False
        if args.num_workers is not None:
            hp_dict['num_workers'] = args.num_workers
        if args.downsample is not None:
            hp_dict['downsample'] = args.downsample
        if args.device is not None:
            device = args.device

    ###################################
    # Create Dataset # Create Dataset #
    ###################################

    # Get list of track paths
    preprocessed_data_dir = os.path.join(top_dir, 'Datasets', 'Preprocessed')

    if dataset_str == 'SWaT':
        track_dir = os.path.join(preprocessed_data_dir, 'SWaT', 'Train', 'Tracks')
        track_paths = [os.path.join(track_dir, track) for track in os.listdir(track_dir)]
    elif dataset_str == 'SMAP':
        track_dir = os.path.join(preprocessed_data_dir, 'SMAP', 'Train', 'Tracks')
        track_paths = [os.path.join(track_dir, track) for track in os.listdir(track_dir)]
    elif dataset_str == 'WADI':
        track_dir = os.path.join(preprocessed_data_dir, 'WADI', 'Train', 'Tracks')
        track_paths = [os.path.join(track_dir, track) for track in os.listdir(track_dir)]
    elif dataset_str == 'MSL':
        track_dir = os.path.join(preprocessed_data_dir, 'MSL', 'Train', 'Tracks')
        track_paths = [os.path.join(track_dir, track) for track in os.listdir(track_dir)]
    elif dataset_str == 'SMD':
        track_dir = os.path.join(preprocessed_data_dir, 'SMD', 'Train', 'Tracks')
        track_paths = [os.path.join(track_dir, track) for track in os.listdir(track_dir)]
    elif dataset_str == 'DSN_1k':
        track_dir = os.path.join(preprocessed_data_dir, 'DSN_1k', 'Train', 'Tracks')
        track_paths = [os.path.join(track_dir, track) for track in os.listdir(track_dir)]
    else:
        raise Exception(f"Invalid dataset_str: {dataset_str}")

    print(f"Loading dataset {dataset_str}")

    # Initialize dataset
    dataset = TSDataset_tracks(track_paths=track_paths, window_size=hp_dict['window_sz'], padding=hp_dict['padding'], downsample=hp_dict['downsample'], scaling=True)
    dataloader = DataLoader(dataset, batch_size=hp_dict['batch_sz'], shuffle=hp_dict['shuffle'], num_workers=hp_dict['num_workers'])
    print(f"Length of dataset:", len(dataset))

    ###############################
    # Create Model # Create Model #
    ###############################

    print(f"Creating model {model_str}")
    if model_str == 'TranAD':
        # Allows creative testing of dim_feedforward, see description of the function
        parsed_dff = helpers.parse_dim_feedforward(hp_dict['features'], hp_dict['dim_feedforward'])
        hp_dict['dim_feedforward'] = parsed_dff

        # Create TranAD model, optimizer, and scheduler
        model = models.TranAD(n_feats = hp_dict['features'], dim_feedforward=parsed_dff, batch_sz = hp_dict['batch_sz'], window_sz = hp_dict['window_sz'], num_encoder_layers=hp_dict['num_layers'], num_decoder_layers=hp_dict['num_layers']).to(device).to(torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr = hp_dict['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    elif model_str == 'USAD':
        # Identify USAD model architecture from inputs
        w_size = hp_dict['window_sz']*hp_dict['features'] # size of window as a vector
        z_size = hp_dict['hidden_sz'] # hidden dimension/ latent space size

        hp_dict['w_size'] = w_size
        hp_dict['z_size'] = z_size

        # Create USAD model and optimizers
        model = models.USAD(w_size, z_size).to(device).to(torch.float32)
        optimizer1 = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=hp_dict['lr'])
        optimizer2 = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=hp_dict['lr'])
    else:
        raise Exception(f"Error: Model {model_str} not supported")

    ###################################
    # Helpful Prints # Helpful Prints #
    ###################################

    print(f"Training model {model_str} on {dataset_str}")
    print(f"Hyperparameters:")
    print_data.print_hyperparameters(hp_dict)

    #######################
    # Training # Training #
    #######################

    start = time.time()
    # Location where we store checkpoints
    model_path = os.path.join(checkpoint_dir, f'{hp_dict["identifier"]}.ckpt')
    if model_str == 'TranAD':
        hp_dict['final_loss'], hp_dict['final_lr'] = helpers.train_TranAD(model=model, epochs=hp_dict['epochs'], dataloader=dataloader, optimizer=optimizer, scheduler=scheduler, device=device, save_location=model_path)
    elif model_str == 'USAD':
        hp_dict['losses'] = helpers.train_USAD(hp_dict['epochs'], model, dataloader, optimizer1, optimizer2, device)
    else:
        raise Exception(f"Error: Model {model_str} not supported")
    end = time.time()
    print(f"Elapsed time: {end-start:0.2f}s")

    ###########################
    # Save Model # Save Model #
    ###########################

    print(f"Saving full model and hyperparameters to {model_path}")

    if model_str == 'TranAD':
        optimizer_states = optimizer.state_dict()
        scheduler_states = scheduler.state_dict()
        model_states = model.state_dict()
    elif model_str == 'USAD':
        optimizer_states = [optimizer1.state_dict(), optimizer2.state_dict()]
        scheduler_states = None
        model_states = model.state_dict()
    else:
        raise Exception(f"Error: Model {model_str} not supported")

    torch.save({
        'hp_dict': hp_dict,
        'model_states': model_states,
        'optimizer_states': optimizer_states,
        'scheduler_states': scheduler_states},
        model_path)

if __name__ == '__main__':
    # To allow num_workers>0
    torch.multiprocessing.set_start_method('spawn', force=True)

    # To allow CLIs
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_str', type=str, default=None)
    parser.add_argument('--dataset_str', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_sz', type=int, default=None)
    parser.add_argument('--window_sz', type=int, default=None)
    parser.add_argument('--hidden_sz', type=int, default=None)
    parser.add_argument('--dim_feedforward', default=None)
    parser.add_argument('--pot_q', type=float, default=None)
    parser.add_argument('--shuffle', type=int, default=None)
    parser.add_argument('--padding', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--downsample', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--identifier', type=str, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--scaling', type=bool, default=None)
    args = parser.parse_args()

    main(args)
