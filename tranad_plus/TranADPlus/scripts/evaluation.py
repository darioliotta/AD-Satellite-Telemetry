###########
# Imports #
###########
print("Loading imports")

import os
from pathlib import Path
import sys
import torch
import numpy as np
import pickle
import argparse
from pprint import pprint
from torch.utils.data import DataLoader

# Update path to include TranADPlus
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

from TranADPlus.src import global_config, models, helpers
from TranADPlus.src.datasets import TSDataset_tracks

#######################################################
# Main Evaluation Function # Main Evaluation Function # 
#######################################################

def main(args=None):
    #############################################
    # Load Saved Metadata # Load Saved Metadata #
    #############################################
    print("Loading saved metadata")

    # Load config
    config = global_config.config

    # Location where we store checkpoints
    experiment_dir = os.path.join(config.top_dir)
    checkpoint_dir = os.path.join(experiment_dir, 'Checkpoints')
    checkpoint_str = os.path.join(checkpoint_dir, f'{args.identifier}.ckpt')
    checkpoint = torch.load(checkpoint_str)
    helpers.make_dir_tree(os.path.join(experiment_dir, "Pickles"))

    # Load data onto GPU if available
    if not torch.cuda.is_available():
        print(f"Warning: CUDA unavailable. Using CPU-only training instead.")
        device = 'cpu'
    else:
        device = 'cuda:0'
    if args.device is not None:
        device = args.device

    # Get hyperparameter dict
    hp_dict = checkpoint['hp_dict']
    dataset_str = hp_dict['dataset_str']
    model_str = hp_dict['model_str']

    ##############
    # Load model #
    ##############
    print(f"Loading model {model_str}")

    if model_str == 'USAD':
        # Identify USAD model architecture from inputs
        w_size = hp_dict['w_size'] # size of window as a vector
        z_size = hp_dict['z_size'] # hidden dimension/ latent space size

        model = models.USAD(w_size, z_size).to(device).to(torch.float32)
        optimizer1 = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=hp_dict['lr'])
        optimizer2 = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=hp_dict['lr'])

        # Create USAD model and optimizers
        model.load_state_dict(checkpoint['model_states'])
        optimizer1.load_state_dict(checkpoint['optimizer_states'][0])
        optimizer2.load_state_dict(checkpoint['optimizer_states'][1])

    elif hp_dict['model_str'] == 'TranAD':
        # Allows creative testing of dim_feedforward, see description of the function
        parsed_dff = helpers.parse_dim_feedforward(hp_dict['features'], hp_dict['dim_feedforward']) 
        model = models.TranAD(n_feats = hp_dict['features'], dim_feedforward=parsed_dff, batch_sz = hp_dict['batch_sz'], window_sz = hp_dict['window_sz'], num_encoder_layers=hp_dict['num_layers'], num_decoder_layers=hp_dict['num_layers']).to(device).to(torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp_dict['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

        # Create TranAD model, optimizer, and scheduler
        model.load_state_dict(checkpoint['model_states'])
        optimizer.load_state_dict(checkpoint['optimizer_states'])
        scheduler.load_state_dict(checkpoint['scheduler_states'])
    else:
        raise Exception(f"Model string {hp_dict['model_str']} invalid.")

    ################
    # Load dataset #
    ################
    print(f"Loading {dataset_str} training and testing datasets")

    # Get list of track paths
    preprocessed_data_dir = os.path.join(config.top_dir, 'Datasets', 'Preprocessed')

    inference_sz = hp_dict['batch_sz']
    if dataset_str == 'WADI':
        train_track_dir = os.path.join(preprocessed_data_dir, 'WADI', 'Train', 'Tracks')
        train_track_paths = [os.path.join(train_track_dir, track) for track in os.listdir(train_track_dir)]

        train_anom_dir = os.path.join(preprocessed_data_dir, 'WADI', 'Train', 'Labels')
        train_anom_paths = [os.path.join(train_anom_dir, anom) for anom in os.listdir(train_anom_dir)]
        
        test_track_dir = os.path.join(preprocessed_data_dir, 'WADI', 'Test', 'Tracks')
        test_track_paths = [os.path.join(test_track_dir, track) for track in os.listdir(test_track_dir)]

        test_anom_dir = os.path.join(preprocessed_data_dir, 'WADI', 'Test', 'Labels')
        test_anom_paths = [os.path.join(test_anom_dir, anom) for anom in os.listdir(test_anom_dir)]

    elif dataset_str == 'SWaT':
        train_track_dir = os.path.join(preprocessed_data_dir, 'SWaT', 'Train', 'Tracks')
        train_track_paths = [os.path.join(train_track_dir, track) for track in os.listdir(train_track_dir)]

        train_anom_dir = os.path.join(preprocessed_data_dir, 'SWaT', 'Train', 'Labels')
        train_anom_paths = [os.path.join(train_anom_dir, anom) for anom in os.listdir(train_anom_dir)]
        
        test_track_dir = os.path.join(preprocessed_data_dir, 'SWaT', 'Test', 'Tracks')
        test_track_paths = [os.path.join(test_track_dir, track) for track in os.listdir(test_track_dir)]

        test_anom_dir = os.path.join(preprocessed_data_dir, 'SWaT', 'Test', 'Labels')
        test_anom_paths = [os.path.join(test_anom_dir, anom) for anom in os.listdir(test_anom_dir)]

    elif dataset_str == 'SMAP':
        train_track_dir = os.path.join(preprocessed_data_dir, 'SMAP', 'Train', 'Tracks')
        train_track_paths = [os.path.join(train_track_dir, track) for track in os.listdir(train_track_dir)]

        train_anom_dir = os.path.join(preprocessed_data_dir, 'SMAP', 'Train', 'Labels')
        train_anom_paths = [os.path.join(train_anom_dir, anom) for anom in os.listdir(train_anom_dir)]
        
        test_track_dir = os.path.join(preprocessed_data_dir, 'SMAP', 'Test', 'Tracks')
        test_track_paths = [os.path.join(test_track_dir, track) for track in os.listdir(test_track_dir)]

        test_anom_dir = os.path.join(preprocessed_data_dir, 'SMAP', 'Test', 'Labels')
        test_anom_paths = [os.path.join(test_anom_dir, anom) for anom in os.listdir(test_anom_dir)]

    elif dataset_str == 'MSL':
        train_track_dir = os.path.join(preprocessed_data_dir, 'MSL', 'Train', 'Tracks')
        train_track_paths = [os.path.join(train_track_dir, track) for track in os.listdir(train_track_dir)]

        train_anom_dir = os.path.join(preprocessed_data_dir, 'MSL', 'Train', 'Labels')
        train_anom_paths = [os.path.join(train_anom_dir, anom) for anom in os.listdir(train_anom_dir)]
        
        test_track_dir = os.path.join(preprocessed_data_dir, 'MSL', 'Test', 'Tracks')
        test_track_paths = [os.path.join(test_track_dir, track) for track in os.listdir(test_track_dir)]

        test_anom_dir = os.path.join(preprocessed_data_dir, 'MSL', 'Test', 'Labels')
        test_anom_paths = [os.path.join(test_anom_dir, anom) for anom in os.listdir(test_anom_dir)]

    elif dataset_str == 'SMD':
        train_track_dir = os.path.join(preprocessed_data_dir, 'SMD', 'Train', 'Tracks')
        train_track_paths = [os.path.join(train_track_dir, track) for track in os.listdir(train_track_dir)]

        train_anom_dir = os.path.join(preprocessed_data_dir, 'SMD', 'Train', 'Labels')
        train_anom_paths = [os.path.join(train_anom_dir, anom) for anom in os.listdir(train_anom_dir)]
        
        test_track_dir = os.path.join(preprocessed_data_dir, 'SMD', 'Test', 'Tracks')
        test_track_paths = [os.path.join(test_track_dir, track) for track in os.listdir(test_track_dir)]

        test_anom_dir = os.path.join(preprocessed_data_dir, 'SMD', 'Test', 'Labels')
        test_anom_paths = [os.path.join(test_anom_dir, anom) for anom in os.listdir(test_anom_dir)]

    elif dataset_str == 'DSN_1k':
        train_track_dir = os.path.join(preprocessed_data_dir, 'DSN_1k', 'Train', 'Tracks')
        train_track_paths = [os.path.join(train_track_dir, track) for track in os.listdir(train_track_dir)]

        train_anom_dir = os.path.join(preprocessed_data_dir, 'DSN_1k', 'Train', 'Labels')
        train_anom_paths = [os.path.join(train_anom_dir, anom) for anom in os.listdir(train_anom_dir)]
        
        test_track_dir = os.path.join(preprocessed_data_dir, 'DSN_1k', 'Test', 'Tracks')
        test_track_paths = [os.path.join(test_track_dir, track) for track in os.listdir(test_track_dir)]

        test_anom_dir = os.path.join(preprocessed_data_dir, 'DSN_1k', 'Test', 'Labels')
        test_anom_paths = [os.path.join(test_anom_dir, anom) for anom in os.listdir(test_anom_dir)]

    else:
        raise Exception(f"Invalid dataset_str: {dataset_str}")

    # Initialize datasets
    train_dataset = TSDataset_tracks(track_paths=train_track_paths, anom_paths=train_anom_paths, window_size=hp_dict['window_sz'], padding=hp_dict['padding'], scaling=True, downsample=hp_dict['downsample'])
    train_dataloader = DataLoader(train_dataset, batch_size=inference_sz, shuffle=False, num_workers=hp_dict['num_workers'])
    print(f"Length of train dataset:", len(train_dataset))

    test_dataset = TSDataset_tracks(track_paths=test_track_paths, anom_paths=test_anom_paths, window_size=hp_dict['window_sz'], padding=hp_dict['padding'], scaling=True, scaler=train_dataset.scaler, downsample=hp_dict['downsample'])
    test_dataloader = DataLoader(test_dataset, batch_size=inference_sz, shuffle=False, num_workers=hp_dict['num_workers'])
    print(f"Length of test dataset:", len(test_dataset))
    print()

    #####################################
    # Use Model to Generate Predictions #
    #####################################
    print("Generating prediciton")
    print()

    # Try different alpha/beta combinations and keep the best in terms of F1 score
    # We save the best F1 for both POT method and BF search method
    if args.alpha_search:
        alphas = np.linspace(0,1,21)
    else:
        alphas = [0.5]
    bf_best_f1 = -np.inf
    bf_best_ret = None
    bf_best_alpha = None
    pot_best_f1 = -np.inf
    pot_best_ret = None
    pot_best_alpha = None
    for alpha in alphas:
        beta = 1.-alpha
        if model_str == 'USAD':
            output = helpers.evaluate_USAD(model, test_dataloader, train_dataloader, alpha=alpha, beta=beta, pot_q=hp_dict['pot_q'], device=device)
        elif model_str == 'TranAD':
            output = helpers.evaluate_TranAD(model, train_dataloader, test_dataloader, alpha=alpha, beta=beta, pot_q=hp_dict['pot_q'], device=device)
        else:
            raise Exception(f"Invalid model_str: {model_str}")
            break
        if output['bf_f1'] > bf_best_f1:
            bf_best_ret = output
            bf_best_f1 = output['bf_f1']
            bf_best_alpha = alpha
        if output['pot_f1'] > pot_best_f1:
            pot_best_ret = output
            pot_best_f1 = output['pot_f1']
            pot_best_alpha = alpha
    bf_best_ret.update({"alpha": bf_best_alpha})
    bf_best_ret.update({"beta": 1-bf_best_alpha})
    print(f"Best bf output had alpha={bf_best_alpha}") 
    pprint(bf_best_ret)
    print()
    pot_best_ret.update({"alpha": pot_best_alpha})
    pot_best_ret.update({"best": 1-pot_best_alpha})
    print(f"Best pot output had alpha={pot_best_alpha}") 
    pprint(pot_best_ret)

    # Save output results as pickles
    fname = hp_dict['identifier'] + '_bf.pkl'
    with open(os.path.join(experiment_dir, 'Pickles', fname), 'wb') as f:
        pickle.dump(bf_best_ret, f)

    fname = hp_dict['identifier'] + '_pot.pkl'
    with open(os.path.join(experiment_dir, 'Pickles', fname), 'wb') as f:
        pickle.dump(pot_best_ret, f)

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
    parser.add_argument('--alpha_search', type=bool, default=False)
    args = parser.parse_args()

    main(args)