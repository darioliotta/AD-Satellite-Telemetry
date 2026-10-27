import torch
from sklearn import preprocessing
import TranADPlus.src.helpers as helpers
from typing import Union

class TSDataset_tracks(torch.utils.data.Dataset):
    def __init__(self, *, window_size: int, track_paths: list[str], anom_paths: list[str] = None, scaling: bool = False, scaler: bool = None, padding: bool = False, downsample: int = None, debug: bool = False) -> None:
        """
        Creates a PyTorch Dataset for preprocessed timeseries data. Loads all tracks into RAM as a single track. When anomaly paths are provided, return those too.  

        `window_size`: length of a single window in the dataset (int)  
        `track_paths`: list of paths for preprocessed tracks (list)  
        `anom_paths`: list of paths for preprocessed anoms (list, optional)  
        `scaling`: If true, enable TranAD scaling (bool, optional)  
        `scaler`: Fitted scaler (used for testing data) (sklearn.preprocessing.MinMaxScaler, optional)  
        `padding`: when True, tracks are padded similar to TranAD (boo, optional)  
        `downsample`: when an integer, downsamples the data by taking the median (USAD) (int, optional)  

        Returns: `None`

        NOT UNIT TESTED  
        Reason: It is very hard to unit test, the functionality is verified just by running it
        """
        # Verify we have the same number of track and anomaly files
        if anom_paths is not None:
            assert(len(track_paths) == len(anom_paths))

        # Save window size
        self.window_size = window_size

        # Save that we used padding
        self.padding = padding

        # Save big track to class
        self.big_track = helpers.merge_tracks(track_paths, window_size, padding)
        if anom_paths is not None:
            self.big_anom = helpers.merge_tracks(anom_paths, window_size, padding)
        else:
            self.big_anom = None

        # Scaling when used and provided
        # Note: For training data we provide the MinMaScaler to this function
        #       to get the scaling values from when we fitted to training data
        if scaling:
            if scaler is None:
                self.scaler = preprocessing.MinMaxScaler()
                self.big_track = self.scaler.fit_transform(self.big_track)
            else:
                self.scaler = scaler
                self.big_track = self.scaler.transform(self.big_track)

        # Down sampling (USAD uses this in their papaer)
        if not downsample is None:
            self.big_track = helpers.downsample(self.big_track, downsample)
            if anom_paths is not None:
                self.big_anom = helpers.downsample(self.big_anom, downsample)

        # Calculate number of data points (unique windows)
        self.data_count = int(self.big_track.shape[0]) - window_size + 1

        # Convert to torch
        self.big_track = torch.from_numpy(self.big_track)
        if anom_paths is not None:
            self.big_anom = torch.from_numpy(self.big_anom)

        return None

    def __len__(self) -> int:
        """
        Returns: `int`
        """
        return self.data_count

    def __getitem__(self, index) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns: `torch.Tensor`
        """
        window = self.big_track[index:index+self.window_size,:]
        if self.big_anom is not None:
            anoms = self.big_anom[index:index+self.window_size,:]
            return window, anoms
        else:
            return window
