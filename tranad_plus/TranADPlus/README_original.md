# TranAD+

This repository contains the base [TranAD](https://github.com/imperial-qore/TranAD) and [USAD](https://github.com/manigalati/usad) models with a custom pipeline built around them for easy training and evaluation and improvements to TranAD. 

## Results

![TranAD+ Comparison On DSN_1k](Results/Media/ResultsTable1.png?raw=true "DSN_1k_Results")
![TranAD+ Comparison On SMAP](Results/Media/ResultsTable2.png?raw=true "SMAP_Results")
![TranAD+ Comparison On MSL](Results/Media/ResultsTable3.png?raw=true "MSL_Results")
![TranAD+ Comparison On SWaT](Results/Media/ResultsTable4.png?raw=true "SWaT_Results")
![TranAD+ Comparison On SMD](Results/Media/ResultsTable5.png?raw=true "SMD_Results")
![TranAD+ Comparison On WADI](Results/Media/ResultsTable6.png?raw=true "WADI_Results")

## Getting Started

To set up the environment, I use the `conda` python package manager, all packages are listed in `environment.yml`. To create the environment, run:
```bash
conda env create -f environment.yml
```

To get familiar with everything, I would head over to the `notebooks` folder and start with `parse_datasets.ipynb` followed by `train_then_evaluate.ipynb`.

## Model Checkpoints

All model results presented in the paper can be found in the `Results` directory.

The model checkpoints presented in the paper for `TranAD+` can be found [here](https://drive.google.com/drive/folders/1CcdcmmBQIWfF84qR62q6kZUvCsf9B8Wb?usp=sharing). Please put the two directories, `Checkpoints` and `Pickles` in the top-level directory of this repo, and run `notebooks/get_best_results.ipynb` to print the run results.

The model checkpoints for `TranAD` can be found [here](https://drive.google.com/drive/folders/1Cr9HK_DkzM_vMJ6HH8RI2BV-fLhSQi5T?usp=drive_link).

## Datasets

- `SMAP`: The `SMAP` dataset can be found [here](https://github.com/khundman/telemanom).
- `MSL`: The `MSL` dataset can be found [here](https://github.com/khundman/telemanom).
- `SWaT`: The `SWaT` dataset can be found [here](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/).
- `WADI`: The `WADI` dataset can be found [here](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/).
- `SMD`: The `SMD` dataset can be found [here](https://github.com/NetManAIOps/OmniAnomaly).
- `DSN_1k`: The `DSN_1k` dataset can be found [here](https://osf.io/t9c2a/?view_only=8d1b376367a94599a8d1ab53253ea2b3).

Datasets raw files should be placed in `Datasets/Raw/<dataset_name>`. These are then copied into `Datasets/Original` with `preprocess_data.initialize_dataset()`. Then, the tracks must be parsed using `parse_data.parse_tracks()`, saving results in `Datasets/EDA`. Afterwards, datasets can be preprocessed using `preprocess_data.preprocess_data()`, storing the preprocessed datasets used for training in `Datasets/Preprocessed`. A notebook demonstrating this is shown in [notebooks/parse_datasets.ipynb](notebooks/parse_dataset.ipynb).

![Dataset Comparison Table](Results/Media/Dataset_Table.png?raw=true "Dataset_Comparison")

## Directory Structure

```text
TranADPlus
├── __init__.py
├── Results
├── Checkpoints
├── Datasets
│   ├── EDA
│   ├── Original
│   ├── Preprocessed
│   └── Raw
├── environment.yml
├── licenses
├── LICENSE
├── notebooks
├── Pickles
├── scripts
├── src
├── test
└── README.md
```

`__init__.py`:
- Used to initialize the package, it is particularly important for getting the global config file from `src/global_config` to be a single reference across the package. If anyone has a better suggestion I'm all-ears.

`Results`:
- Runs and results for the other models presented in the paper, saved for transparency. Also contains tables presented in the paper.

`Checkpoints`:
- Folder generated when running `notebooks/train_then_evaluate.ipynb` and/or `scripts/training_loop.py`. Contains model checkpoints.

`Datasets`:
- Contains all dataset-related information. This includes the `Raw` datasets, the `Original` folder which has all datasets in a common format, information for `EDA`, and the `Preprocessed` datasets.

`environment.yml`:
- Conda environment packages and their versions.

`licenses`:
- Contains the licenses for TranAD+, TranAD, OmniAnomaly, and USAD.

`LICENSE`:
- Contains the license for TranAD+ (used for GitHub page UI niceness).

`notebooks`:
- Contains helpful notebooks.
- `parse-datasets.ipynb`: Shows how to initialize datasets.
- `train_then_evaluate.ipynb`: Shows how to train and evaluate a model.
- `pot.ipynb`: Shows how POT works qualitatively.
- `get_best_results.ipynb`: Takes in the final `Checkpoints` and `Pickles` to collect the best F1 score for each (model, dataset) combination.

`Pickles`:
- Folder generated when running `notebooks/train_then_evaluate.ipynb` and/or `scripts/evaluation.py`. Contains the final result pickle files with F1 scores.

`scripts`:
- Contains training and evaluation scripts (`training_loop.py` and `evaluation.py`). Also contains `clean_data.py` which can help with removing files from failed runs.

`src`:
- Contains the `TranADPlus` package.

`test`:
- Unit tests for the majority of the `TranADPlus` package.

`README.md`:
- This file

## Unit Testing

Most function in the `src` directory have been unit tested. To run the following, make sure you're in the top-most directory for `TranADPlus`.

Run all tests:
- `python -m unittest -b`

Run specific test:
- `python -m unittest test.test_print_data.TestPrintData.test_print_nan_info`

## Using custom datasets

To use a custom dataset, you need to include an initialization function in `preprocess_data.initialize_dataset()`. This puts your custom dataset into a standard format and directory structure recognized by the rest of the functions in this package.

## Changes from TranAD

- We switch out the feed-forward component of the model to be an inverse bottleneck instead of a bottleneck.
- We re-implement USAD's $\alpha$ and $\beta$ scaling on the decoder outputs: $\mathcal{A} = \alpha D_1 + \beta D_2$. The original TranAD paper assigned $\alpha=\beta=0.5$ and the code had $\alpha=0$, $\beta=1$.
- Resolved unaddressed issue in [TranAD](https://github.com/imperial-qore/TranAD) code where first window is repeated exactly.
- Resolved unaddressed issue in [TranAD](https://github.com/imperial-qore/TranAD) where `SMD`, `SMAP`, and `MSL` datasets were trained on a single track instead of the whole dataset.
- Training hyperparameters were optimized for better results.
- Improved experiment reproducability by providing model weights and a specific Python environment.

## Other models

I was also able to get [MTAD-GAT](https://github.com/ML4ITS/mtad-gat-pytorch) and [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) up and running, but didn't include them in this paper since their respective GitHub pages have well-written training pipelines. OmniAnomaly was hard to set-up due to using an old verison of TensorFlow. Don't hesitate to reach out if you need some help with those repos.

## Licenses

TranAD+ follows the BSD 3-Clause license. See [here](licenses/TranAD+.md).  
TranAD follows the BSD 3-Clause license. See [here](licenses/TranAD.md).  
USAD follows the BSD 3-Clause license. See [here](licenses/USAD.md).  
OmniAnomaly follows the MIT license. See [here](licenses/OmniAnomaly.md).  

## Contributors

This repository was created by Alexey Yermakov.

## Citing

This repository:

```bibtex
@misc{Yermakov2024tranadplus,
  author = {Yermakov, Alexey},
  title = {TranAD+},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yyexela/TranADPlus}}
}
```

The original TranAD paper:

```bibtex
@article{tuli2022tranad,
  title={{TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data}},
  author={Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R},
  journal={Proceedings of VLDB},
  volume={15},
  number={6},
  pages={1201-1214},
  year={2022}
}
```

The original USAD paper:

```bibtex
@inproceedings{audibert2020usad,
  title={Usad: Unsupervised anomaly detection on multivariate time series},
  author={Audibert, Julien and Michiardi, Pietro and Guyard, Fr{\'e}d{\'e}ric and Marti, S{\'e}bastien and Zuluaga, Maria A},
  booktitle={Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery \& data mining},
  pages={3395--3404},
  year={2020}
}
```
