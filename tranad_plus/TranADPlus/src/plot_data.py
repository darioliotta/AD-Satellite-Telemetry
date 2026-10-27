###############################
# Imports # Imports # Imports #
###############################

# System
import numpy as np
import pandas as pd
import torch
from typing import Union, Literal
from TranADPlus.src.helpers import flatten_list, get_dataset_dfs
from TranADPlus.src.preprocess_data import preprocess_plotly
import TranADPlus.src.global_config as global_config

# Load config
config = global_config.config

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

################################################################
# Plotting functions # Plotting functions # Plotting functions #
################################################################

def add_timeseries_data(fig: go.Figure, data: Union[pd.DataFrame, np.ndarray, torch.Tensor],\
                        plot_params: list[int] = None, indices: list[int] = None, color: str = None, width: int = None) -> go.Figure:
    """
    Given an existing subplot figure, add the parameter data from the provided data to the subplots.

    `data` can be a dataframe or from a numpy array.
    `plot_params` is a list of column indices for which parameters to plot.
    `indices` is a list of indices for which sub-plots to plot the parameters on. Must be the same shape as `plot_params`
    `color` is the color to plot the parameters as

    Returns: `go.Figure`

    NOT UNIT TESTED  
    Reason: Informational only, not critical for functionality 
    """
    # Get number of parameters total and pandas column names
    n_params = data.shape[1]

    # If indices is none, just plot parameters in order
    if indices is None:
        indices = list(range(len(plot_params)))

    # Add each parameter sequentially
    for i, param in list(zip(indices,plot_params)):
        if type(data) == pd.DataFrame:
            # Assuming that first column is the time
            param_x = data.index
            param_y = data.iloc[:,param]
        elif type(data) == np.ndarray or type(data) == torch.Tensor:
            # Assuming that data is preprocessed and first column is no longer the time
            param_x = list(range(data.shape[0]))
            param_y = data[:,param]
        else:
            raise Exception(f"Encountered unknown data type: \"{type(data)}\".")

        if param > n_params or param < 0:
            raise Exception(f"Trying to plot parameter {param} for timeseries with {n_params} parameters.")
        fig.add_trace(
            go.Scatter(x=param_x,
                       y=param_y,
                       line=dict(color=color, width=width)),
            row=i+1,
            col=1,
        )
    
    return fig

def gen_timeseries_fig(plot_params: list[int] = None, title: str = "") -> go.Figure:
    """
    Given a list of parameter indices to plot and a title, create a skeleton subplot figure.

    `plot_params`: List of integers for which parameters we will be plotting, used to create subplots (list(int))  
    `title`: Title of figure (str)  

    Returns: `go.Figure`
    
    NOT UNIT TESTED  
    Reason: Informational only, not critical for functionality 
    """
    # Generate subplots
    fig = make_subplots(rows=len(plot_params),
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=[f"Parameter {i}" for i in plot_params])

    # Plot configuration
    fig.update_layout(height=300*len(plot_params),
                      width=1000,
                      title_text=title,
                      showlegend=False)
    fig.update_xaxes(showgrid=False,
                     showticklabels=True,
                     ticks="outside",
                     title="Time")
    fig.update_yaxes(gridcolor='black',
                     griddash="dot",
                     ticks="outside",
                     title="Value")

    return fig

def plot_preprocessed_track_params(track: np.ndarray,\
                                   labels: list,\
                                   plot_params: list[int],\
                                   title: str = "") -> None:
    """
    Given a preprocessed numpy track and (optionally) anomaly windows in labels, plot the
    parameters specified in plot_params (indices in track) with a title

    `track`: Parsed dataset data which we'll be plotting (str)  
    `labels`: List of list of integers containing labeled anomaly regions (str)  
    `plot_params`: list of integers for which parameters to plot, if `None`, plot them all (list(int))  
    `title`: Title of plot (str)  

    Returns: `None`

    NOT UNIT TESTED  
    Reason: Informational only, not critical for functionality 
    """
    # If plot_params is none, plot all parameters
    if plot_params is None:
        plot_params = list(range(track.shape[1]))

    # Make plots
    fig = gen_timeseries_fig(plot_params, title=title)
    fig = add_timeseries_data(fig, track, plot_params)

    # Add labels
    for label in labels:
        fig.add_vrect(label[0], label[1],fillcolor='yellow',opacity=.4,line_width=0,layer='below')

    fig.show()
    return None

def plot_track_params(dataset: Literal['DSN_100', 'DSN_1k', 'SMAP', 'WADI', 'SWaT', 'MSL'], track_name: str, plot_params: list[int] = None, split: str = "Both") -> None:
    """
    Given a track name for a dataset, plot the specified track and parameters from 

    `dataset`: Name of dataset to plot (str)  
    `track_name`: Name of track to plot (str)  
    `plot_params`: list of integers for which parameters to plot, if `None`, plot them all (list(int))  
    `split`: Which split to plot for, defaults to `Both` (str)  

    Returns: `None`
    
    NOT UNIT TESTED  
    Reason: Informational only, not critical for functionality 
    """
    if dataset not in ['DSN_100', 'DSN_1k', 'SMAP', 'WADI', 'SWaT', 'MSL']:
        raise Exception(f"Dataset {dataset} not valid.")

    # Get and preprocess track
    tracks, labels = get_dataset_dfs(dataset, track_names=[track_name], split=split)
    track = tracks[0]
    track_labels = labels[0]
    track= preprocess_plotly(track)
    
    title = f"{dataset} track {track_name}_{split} Parameters"

    plot_preprocessed_track_params(track, track_labels, plot_params, title)
    return None

def plot_predictions(true_y: np.ndarray, pred_y: np.ndarray, labels: list[list[int, int]], anom_score: np.ndarray = None, plot_params: list[int] = None,\
                     line_width: int = None, extra_lines: list[np.ndarray] = []):
    """
    Given true data, predicted data, and a list of anomalies, make a plot of the two.
    If anomaly scores are provided, every other plot is the anomaly score.

    Returns: `None`

    NOT UNIT TESTED  
    Reason: Informational only, not critical for functionality 
    """
    # If plot_params is none, plot all parameters
    if plot_params is None:
        plot_params = list(range(true_y.shape[1]))

    # Adjust for anom_score being provided
    print(f"anom_score type {type(anom_score)}")
    if anom_score is not None:
        plot_indices = list(range(0,2*len(plot_params),2))
        anom_indices = list(range(1,2*len(plot_params),2))
        plot_params_adjusted = flatten_list([[i, i] for i in plot_params])
        title=f"Plot of true and predicted time series data with anomaly scores"
    else:
        plot_indices = list(range(0,len(plot_params),1))
        plot_params_adjusted = plot_params
        title=f"Plot of true and predicted time series data"

    # Make plots
    fig = gen_timeseries_fig(plot_params_adjusted, title=title)
    fig = add_timeseries_data(fig, true_y, plot_params, indices=plot_indices, color='black', width=line_width)
    fig = add_timeseries_data(fig, pred_y, plot_params, indices=plot_indices, color='red', width=line_width)

    # Plot anomalies
    if anom_score is not None:
        fig = add_timeseries_data(fig, anom_score, plot_params, indices=anom_indices, color='green', width=line_width)
        colors = ['purple', 'blue']
        for i in range(len(extra_lines)):
            fig = add_timeseries_data(fig, extra_lines[i], plot_params, indices=anom_indices, color=colors[i], width=line_width)

    # Add labels
    for label in labels:
        fig.add_vrect(label[0], label[1],fillcolor='yellow',opacity=.4,line_width=0,layer='below')

    fig.show()
    return None

def make_pot_plot(train_data, test_data, t_upper=None, t_lower=None, upper_threshold=None, lower_threshold=None):
    """
    Helper function for plotting POT plots from `pot.ipynb`

    `train_data`: Training data that POT uses to get initial threshold (n,) (np.ndarray)  
    `test_data`: Testing data that POT uses to get updated threshold (m,) (np.ndarray)  
    `t_upper`: Initial threshold computed from training data and upper quartile value (float)  
    `t_lower`: Initial threshold computed from training data and lower quartile value (float)  
    `upper_threshold`: Upper threshold to plot (threshold above which values are anomalous) (n,) (np.ndarray)  
    `lower_threshold`: Lower threshold to plot (threshold below which values are anomalous) (m,) (np.ndarray)  

    Returns: `None`

    NOT UNIT TESTED  
    Reason: Informational only, not critical for functionality 
    """
    # Create figure
    fig = make_subplots(rows=1, cols=1)
    # Plot training data
    fig.add_trace(
        go.Scatter(y=train_data,
                name="Train Data",
                line_color='green'),
        row=1, col=1
    )
    # Plot testing data
    fig.add_trace(
        go.Scatter(y=test_data,
                x=np.arange(train_data.shape[0],train_data.shape[0]+test_data.shape[0]),
                name="Test Data",
                line_color='purple'),
        row=1, col=1
    )
    # Add anomaly thresholds and t
    names = []
    values = []
    colors = []
    if t_upper is not None:
        names.append("t_upper")
        values.append(np.ones(train_data.shape[0]+test_data.shape[0])*t_upper)
        colors.append('black')
    if t_lower is not None:
        names.append("t_lower")
        values.append(np.ones(train_data.shape[0]+test_data.shape[0])*t_lower)
        colors.append('black')
    if upper_threshold is not None:
        names.append("Upper Threshold")
        values.append(upper_threshold)
        colors.append('red')
    if lower_threshold is not None:
        names.append("Lower Threshold")
        values.append(lower_threshold)
        colors.append('red')
    # Plot thresholds
    for name, value, color in zip(names, values, colors):
        fig.add_trace(
            go.Scatter(y=value,
                name=name,
                line_color=color),
            row=1, col=1
        )
    fig.show()
    return None