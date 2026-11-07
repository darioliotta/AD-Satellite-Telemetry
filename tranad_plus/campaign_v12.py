"""TOAST v12: complete the v10 4x4 architectural grids to 6x5, with no repeats.

For each dataset the v12 campaign trains exactly the set difference:
  ((old W + 2 new W) x (old FF + 1 new FF)) - (old W x old FF).
Thus 14 new architectures x 3 seeds per dataset = 126 runs, 50 epochs.
"""
from __future__ import annotations

BASES = {
    'OPS-SAT': {'base': 'opssat_shared_ws10_1f', 'features': 1, 'window': 10,
                'feedforward': 8, 'batch': 256, 'lr': 0.001, 'pot_q': 0.01},
    'SMAP': {'base': 'smap_plus_ws10', 'features': 25, 'window': 10,
             'feedforward': 50, 'batch': 1024, 'lr': 0.001, 'pot_q': 0.07},
    'MSL': {'base': 'msl_plus_ws50_dff2x', 'features': 55, 'window': 50,
            'feedforward': 110, 'batch': 128, 'lr': 0.0001, 'pot_q': 0.01},
}
# The v10 effective INTEGER dimensions, derived from 25/50/75/100%.
HISTORICAL = {
    'OPS-SAT': {'windows': (3, 5, 8, 10), 'feedforwards': (2, 4, 6, 8)},
    'SMAP': {'windows': (3, 5, 8, 10), 'feedforwards': (13, 25, 38, 50)},
    'MSL': {'windows': (13, 25, 38, 50), 'feedforwards': (28, 55, 83, 110)},
}
ADDITIONS = {
    'OPS-SAT': {'windows': (4, 6), 'feedforwards': (3,)},
    'SMAP': {'windows': (4, 6), 'feedforwards': (32,)},
    'MSL': {'windows': (32, 44), 'feedforwards': (69,)},
}
GRID = {
    name: {
        'windows': tuple(sorted((*HISTORICAL[name]['windows'], *ADDITIONS[name]['windows']))),
        'feedforwards': tuple(sorted((*HISTORICAL[name]['feedforwards'], *ADDITIONS[name]['feedforwards']))),
    } for name in BASES
}
SEEDS = (2026, 2027, 2028)
EPOCHS = 50


def new_pairs(dataset):
    all_pairs = {(w, f) for w in GRID[dataset]['windows'] for f in GRID[dataset]['feedforwards']}
    old_pairs = {(w, f) for w in HISTORICAL[dataset]['windows'] for f in HISTORICAL[dataset]['feedforwards']}
    assert len(all_pairs) == 30 and len(old_pairs) == 16 and old_pairs <= all_pairs
    return tuple(sorted(all_pairs - old_pairs))


def identifier(dataset: str, window: int, feedforward: int, seed: int) -> str:
    return f"toast12_{dataset.lower().replace('-', '')}_w{window:03d}_h{feedforward:03d}_s{seed}"


def experiments():
    items = []
    for dataset, baseline in BASES.items():
        for window, ff in new_pairs(dataset):
            for seed in SEEDS:
                entry = {
                    'base_profile': baseline['base'],
                    'identifier': identifier(dataset, window, ff, seed),
                    'epochs': EPOCHS, 'batch_sz': baseline['batch'],
                    'window_sz': window, 'dim_feedforward': ff,
                    'seed': seed,
                    'ablation_family': 'toast_v12_completed_v10_grid',
                    'ablation_factor': 'combined',
                    'ablation_level': f'w{window:03d}_h{ff:03d}',
                    # For plots/reporting only; never reuse v10 checkpoint.
                    'ablation_reference': f"toast10_{dataset.lower().replace('-', '')}_w100_h100_s{seed}",
                }
                if dataset == 'OPS-SAT':
                    # IDENTICAL v10 protocol: 20% nominal validation by channel,
                    # all official-train anomalous fragments in validation;
                    # fixed split_seed, untouched official test.
                    entry.update({'segments_csv': '@data/raw/OPSSAT/segments.csv',
                                  'validation_fraction': 0.2, 'split_seed': 2026})
                items.append(entry)
    return items


def make_plan():
    return {
        'schema_version': 1,
        'description': 'TOAST v12: 126 new 50-epoch training runs (42/dataset) completing v10 4x4 to 6x5; 3 seeds; v10-aligned per-dataset splits; test-label-optimized metrics are oracle',
        'experiments': experiments(),
        'measurement': {
            'enabled': False, 'warmup': 10, 'iterations': 50,
            'score_batch_size': 1024, 'alpha': 0.5,
            'threshold_train_quantile': 0.995,
            'save_raw_scores': True, 'export_decoder_arrays': False,
            'hardware_json_dir': None, 'paper_alpha_search': True,
            'inference_ablation': False,
        },
    }
