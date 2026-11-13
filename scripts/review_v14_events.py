#!/usr/bin/env python3
"""Read-only signal/label/score examples from the completed V14 experiment.

Selection is deterministic at seed 2026 and epoch 100: median-length candidates
for detected anomalies, missed anomalies and false-positive normal intervals.
No weights are loaded; no threshold is fitted using test labels.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'toast-event-review-mpl'))
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tranad_plus.campaigns.exploration.metrics import anomaly_segments, real_point_mask

DATASETS = ('OPS-SAT', 'SMAP', 'MSL')
MODELS = ('reference', 'encoder_only')
NAMES = {'reference': 'TranAD+', 'encoder_only': 'Encoder-only'}
COLORS = {'reference': '#333333', 'encoder_only': '#9B0014'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(project, dataset, seed):
    root = project / 'results/v14/continuation' / dataset
    profiles, scores, thresholds, inputs, labels = {}, {}, {}, [], None
    for model in MODELS:
        folder = root / model / 'reference' / f'seed_{seed}'
        report = json.loads((folder / 'metrics.json').read_text())
        assert report['status'] == 'complete' and report['profile']['epochs'] == 100
        assert sha(folder / 'scores.npz') == report['artifacts']['scores.npz']
        profiles[model] = report['profile']
        with np.load(folder / 'scores.npz', allow_pickle=False) as z:
            row = int(np.flatnonzero(z['alpha_values'] == .5)[0]) if model == 'reference' else 0
            scores[model] = z['test_scores'][row].copy()
            thresholds[model] = float(np.quantile(z['train_scores'][row], .995))
            assert thresholds[model] > 0 and np.isfinite(scores[model]).all()
            if labels is None:
                labels = z['test_labels'].copy()
                ids = z['test_segment_ids'].copy() if dataset == 'OPS-SAT' else None
            else:
                np.testing.assert_array_equal(labels, z['test_labels'])
                if dataset == 'OPS-SAT':
                    np.testing.assert_array_equal(ids, z['test_segment_ids'])
        inputs.extend({'file': str(folder.relative_to(project) / name), 'sha256': sha(folder / name)}
                      for name in ('metrics.json', 'scores.npz'))
    assert profiles['reference']['window_sz'] == profiles['encoder_only']['window_sz']
    profile = profiles['reference']
    # Check the dataset files against the already-verified experiment inventory.
    for entry in report['dataset_inventory']['files']:
        path = project / 'data' / entry['file']
        assert sha(path) == entry['sha256'], path
        inputs.append({'file': str(path.relative_to(project)), 'sha256': entry['sha256']})
    units = []
    if dataset == 'OPS-SAT':
        frame = pd.read_csv(project / 'data/raw/OPSSAT/segments.csv')
        groups = {int(i): g for i, g in frame.groupby('segment', sort=False)}
        split = report['dataset_inventory']['split']
        nominal_training = frame[frame.segment.isin(split['train_fragment_ids'])]
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1])+1]
        ends = np.r_[starts[1:], len(ids)]
        for begin, end in zip(starts, ends):
            segment = int(ids[begin])
            group = groups[segment]
            assert len(group) == end-begin
            np.testing.assert_array_equal(labels[begin:end], group.anomaly.to_numpy())
            dates = pd.to_datetime(group.timestamp, utc=True)
            elapsed = (dates-dates.iloc[0]).dt.total_seconds().to_numpy()
            assert np.all(np.diff(elapsed) >= 0), segment
            channel = str(group.channel.iloc[0])
            nominal = nominal_training[nominal_training.channel == channel].value
            units.append({'id': str(segment), 'channel': channel,
                          'x': elapsed, 'values': group.value.to_numpy(), 'labels': labels[begin:end],
                          'scores': {m: scores[m][begin:end] for m in MODELS},
                          'training_range': (float(nominal.min()), float(nominal.max())),
                          'command_changes': np.array([], dtype=int), 'axis': 'Seconds from fragment start'})
    else:
        parent = project / 'data/processed/tranad_plus' / dataset
        training = [np.load(p, mmap_mode='r', allow_pickle=False) for p in sorted((parent / 'Train/Tracks').glob('*.npy'))]
        training_range = (min(float(a[:, 0].min()) for a in training), max(float(a[:, 0].max()) for a in training))
        mask = real_point_mask(profile, project / 'data', len(labels))
        labels = labels[mask]
        scores = {m: s[mask] for m, s in scores.items()}
        offset = 0
        for path in sorted((parent / 'Test/Tracks').glob('*.npy')):
            values = np.load(path, allow_pickle=False)
            y = (np.load(parent / 'Test/Labels' / path.name).sum(axis=1)>0).astype(np.int8)
            n = len(values)
            np.testing.assert_array_equal(labels[offset:offset+n], y)
            changes = np.flatnonzero(np.any(np.diff(values[:, 1:], axis=0) != 0, axis=1))+1
            units.append({'id': path.stem, 'channel': path.stem, 'x': np.arange(n),
                          'values': values[:, 0], 'labels': y,
                          'scores': {m: s[offset:offset+n] for m, s in scores.items()},
                          'training_range': training_range, 'command_changes': changes,
                          'axis': 'Sample within telemetry track'})
            offset += n
        assert offset == len(labels)
    return units, thresholds, profile, inputs


def candidates(dataset, units, thresholds):
    result = []
    for ui, unit in enumerate(units):
        labels = unit['labels'].astype(bool)
        pred = {m: unit['scores'][m] > thresholds[m] for m in MODELS}
        a, b = anomaly_segments(labels)
        for begin, end in zip(a, b):
            hits = {m: int(pred[m][begin:end].sum()) for m in MODELS}
            result.append({'dataset': dataset, 'unit_index': ui, 'unit': unit['id'],
                           'start': int(begin), 'end_exclusive': int(end), 'length': int(end-begin),
                           'kind': 'anomaly', 'reference_hits': hits['reference'], 'encoder_only_hits': hits['encoder_only'],
                           'both_detected': all(hits.values()), 'both_missed': not any(hits.values()),
                           'normal_begin': 0, 'normal_end': len(labels)})
        if dataset == 'OPS-SAT':
            if not labels.any() and any(pred[m].any() for m in MODELS):
                result.append({'dataset': dataset, 'unit_index': ui, 'unit': unit['id'], 'start': 0,
                               'end_exclusive': len(labels), 'length': len(labels), 'kind': 'false_positive',
                               'reference_hits': int(pred['reference'].sum()), 'encoder_only_hits': int(pred['encoder_only'].sum()),
                               'both_detected': False, 'both_missed': False, 'normal_begin': 0, 'normal_end': len(labels)})
        else:
            normal_a, normal_b = anomaly_segments(~labels)
            fp = (~labels) & (pred['reference'] | pred['encoder_only'])
            fa, fb = anomaly_segments(fp)
            for begin, end in zip(fa, fb):
                group = int(np.searchsorted(normal_a, begin, side='right')-1)
                result.append({'dataset': dataset, 'unit_index': ui, 'unit': unit['id'],
                               'start': int(begin), 'end_exclusive': int(end), 'length': int(end-begin), 'kind': 'false_positive',
                               'reference_hits': int(pred['reference'][begin:end].sum()),
                               'encoder_only_hits': int(pred['encoder_only'][begin:end].sum()),
                               'both_detected': False, 'both_missed': False,
                               'normal_begin': int(normal_a[group]), 'normal_end': int(normal_b[group])})
    return result


def choose(catalog):
    selected = []
    for case in ('detected', 'missed', 'false_positive'):
        pool = [r for r in catalog if (r['kind'] == 'anomaly' and r['both_detected'])] if case == 'detected' else (
               [r for r in catalog if r['kind'] == 'anomaly' and r['both_missed']] if case == 'missed' else
               [r for r in catalog if r['kind'] == 'false_positive'])
        fallback = False
        if not pool and case == 'detected':
            pool = [r for r in catalog if r['kind'] == 'anomaly' and not r['both_missed']]
            fallback = True
        if not pool:
            continue
        median = float(np.median([r['length'] for r in pool]))
        winner = min(pool, key=lambda r: (abs(r['length']-median), r['unit'], r['start']))
        selected.append({**winner, 'case': case, 'eligible_candidates': len(pool),
                         'candidate_median_length': median, 'fallback_any_model_detects': fallback})
    return selected


def adjusted(labels, pred):
    output = pred.copy()
    for a, b in zip(*anomaly_segments(labels)):
        if pred[a:b].any():
            output[a:b] = True
    return output


def draw(dataset, unit, item, thresholds, profile, seed, out):
    n, begin, end = len(unit['labels']), item['start'], item['end_exclusive']
    padding = max(30, min((end-begin)//2, 200))
    left, right = (0, n) if dataset == 'OPS-SAT' else (max(0, begin-padding), min(n, end+padding))
    if item['case'] == 'false_positive' and dataset != 'OPS-SAT':
        left, right = max(left, item['normal_begin']), min(right, item['normal_end'])
    section = slice(left, right)
    x, values, labels = (unit[k][section] for k in ('x', 'values', 'labels'))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 1.35]})
    fig.subplots_adjust(left=.13, right=.97, bottom=.18, top=.86, hspace=.14)
    case_names = {'detected': 'Anomaly with a threshold crossing', 'missed': 'Anomaly without a threshold crossing',
                  'false_positive': 'Normal label with false alarms'}
    fig.suptitle(f"{dataset} | {unit['channel']}" + (f" | fragment {unit['id']}" if dataset == 'OPS-SAT' else '')
                 + '\n' + case_names[item['case']], x=.13, ha='left', fontsize=17, color='#9B0014', weight='bold')
    fig.text(.13, .885, f"Epoch 100 | seed {seed} | W={profile['window_sz']}, FF={profile['dim_feedforward']} | fixed scoring, train-only thresholds", fontsize=9, color='#555555')
    axes[0].plot(x, values, color='#333333', lw=1, marker='.' if dataset == 'OPS-SAT' else None, markersize=2)
    lo, hi = unit['training_range']
    # Avoid expanding the display to training limits outside this local view.
    if lo < values.max() and hi > values.min():
        axes[0].axhspan(max(lo, float(values.min())), min(hi, float(values.max())), color='#BBBBBB', alpha=.15)
    axes[0].set_ylabel('Telemetry value\n(provided scale)' if dataset != 'OPS-SAT' else 'Telemetry value\n(raw CSV scale)')
    if dataset != 'OPS-SAT':
        changes = unit['command_changes']
        changes = changes[(changes >= left) & (changes < right)]
        axes[0].plot(unit['x'][changes], np.full(len(changes), .98), '|', color='#888888', markersize=7,
                     transform=axes[0].get_xaxis_transform(), label='Change in command inputs')
        if len(changes):
            axes[0].legend(loc='upper right', fontsize=8)
    for ax in axes[:2]:
        for a, b in zip(*anomaly_segments(labels)):
            xx0 = x[a]
            xx1 = x[b] if b < len(x) else x[-1]
            if xx1 == xx0:
                xx1 = xx0 + (np.median(np.diff(x)) if len(x)>1 else 1)
            ax.axvspan(xx0, xx1, color='#9B0014', alpha=.10)
        ax.grid(alpha=.15)
    for model in MODELS:
        axes[1].plot(x, np.maximum(unit['scores'][model][section]/thresholds[model], 1e-8),
                     color=COLORS[model], label=NAMES[model], lw=1.15)
    axes[1].axhline(1, color='#888888', linestyle='--', lw=.9)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Score / train threshold\n(log scale; alarm > 1)')
    axes[1].legend(loc='upper right', fontsize=9)
    row_names = []
    for model, raw_row in (('reference', 3), ('encoder_only', 1)):
        pred = unit['scores'][model] > thresholds[model]
        pa = adjusted(unit['labels'], pred)
        for row, mask, name in ((raw_row, pred, NAMES[model]+': direct'), (raw_row-1, pa, NAMES[model]+': PA credit')):
            axes[2].hlines(row, x[0], x[-1], color='#DDDDDD', lw=.6)
            axes[2].vlines(x[mask[section]], row-.3, row+.3, color=COLORS[model], lw=.8)
            row_names.append((row, name))
    axes[2].set_yticks([r for r, _ in sorted(row_names)], [name for _, name in sorted(row_names)], fontsize=8)
    axes[2].set_ylim(-.55, 3.55)
    axes[2].set_xlabel(unit['axis'])
    if x[0] != x[-1]:
        axes[2].set_xlim(x[0], x[-1])
    foot = f"Selected {'fragment' if dataset == 'OPS-SAT' else 'interval'}: {end-begin} samples | "
    foot += f"Direct crossings: TranAD+ {item['reference_hits']}, encoder-only {item['encoder_only_hits']}."
    label_note = ('Whole-fragment label: the true anomaly onset/offset is unknown.' if dataset == 'OPS-SAT' else
                  'Red: annotated anomaly intervals. Top trace: telemetry feature 0; gray ticks: command-input changes.')
    fig.text(.13, .105, foot, fontsize=9)
    fig.text(.13, .078, label_note, fontsize=9, color='#555555')
    fig.text(.13, .05, 'PA credit is a label-based evaluation transformation. Lines use fixed alpha=0.5 for TranAD+; single output for encoder-only.', fontsize=8, color='#666666')
    name = dataset.lower() + '_' + item['case']
    fig.savefig(out / 'figures' / (name+'.pdf'))
    fig.savefig(out / 'figures' / (name+'.png'), dpi=180)
    return fig, {'display_begin': left, 'display_end_exclusive': right,
                 'reference_threshold': thresholds['reference'], 'encoder_only_threshold': thresholds['encoder_only']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    project = args.project_root.resolve()
    out = args.output_root or project / 'results/v14/event_review'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    catalog, selected, inputs, counts = [], [], [], {}
    with PdfPages(out / 'events.pdf') as pdf:
        for dataset in DATASETS:
            units, thresholds, profile, sources = load_dataset(project, dataset, args.seed)
            records = candidates(dataset, units, thresholds)
            choices = choose(records)
            counts[dataset] = {'anomalous_intervals_or_fragments': sum(r['kind']=='anomaly' for r in records),
                               'both_detected': sum(r['both_detected'] for r in records),
                               'both_missed': sum(r['both_missed'] for r in records),
                               'false_positive_candidates': sum(r['kind']=='false_positive' for r in records),
                               'pages': len(choices)}
            for item in choices:
                fig, extra = draw(dataset, units[item['unit_index']], item, thresholds, profile, args.seed, out)
                pdf.savefig(fig)
                plt.close(fig)
                selected.append({**item, **extra, 'seed': args.seed, 'epochs': 100,
                                 'window': profile['window_sz'], 'ff': profile['dim_feedforward']})
            catalog.extend(records)
            inputs.extend(sources)
            print(dataset, counts[dataset], flush=True)
    csv_write(out / 'candidates.csv', catalog)
    csv_write(out / 'selection.csv', selected)
    (out / 'provenance.json').write_text(json.dumps({'seed': args.seed, 'epochs': 100, 'counts': counts,
              'selection': 'Median-length candidate; tie by unit ID and start. No best-seed or model-win selection.',
              'scoring': 'Fixed alpha=.5 for reference; single output for encoder-only; own train-score quantile .995.',
              'threshold_warning': 'Test false-positive rates can differ: the same train quantile is not a matched-test-FPR constraint.',
              'weak_labels': 'OPS-SAT fragment labels repeated on points for display only; onset/offset remains unknown.',
              'code_sha256': sha(Path(__file__)), 'inputs': inputs}, indent=2))
    (out / 'README.md').write_text(
        '# V14 — real event review\n\nOpen `events.pdf` (three examples per dataset when available). Individual vector PDFs and PNGs are in `figures/`.\n\n'
        'Seed 2026, epoch 100, reference configuration, both architectures. Scores use fixed alpha=0.5 for TranAD+ and a single output for encoder-only. '
        'Each model threshold is its own 99.5th train-score percentile. No test-label threshold selection; this does not impose identical test false-positive rates. '
        'The examples illustrate this fixed diagnostic rule, not the independently optimized POT/BF protocols or validation-selected OPS-SAT protocol.\n\n'
        'Examples are the median-length eligible detected anomaly, missed anomaly and false-positive normal interval. '
        'Detection means at least one direct crossing, not correct recognition of all anomalous points. All candidates and the exact selection are exported. '
        'If no anomaly is detected by both, the detected example can use either model; that fallback is explicitly recorded.\n\n'
        'SMAP/MSL: the signal is the first feature in the supplied scale, not recovered engineering units. Gray ticks indicate changes in command inputs, not proven causes. '
        'Padded target duplicates are removed and scores are aligned with real track points. '
        'OPS-SAT: full fragments, original values and elapsed timestamp seconds; the fragment label does not locate the anomaly in time. '
        'Gray signal background indicates the overlapping training value range. PA is illustrated within the annotated intervals of the displayed track/fragment. '
        'These nine examples are qualitative checks; the full-test aggregate results remain the quantitative evidence.\n\n'
        'Reproduce from the project root: `python -B scripts/review_v14_events.py`. No GPU or new training is needed.\n')
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    main()
