#!/usr/bin/env python3
"""Plot the observed v10/v12 grid without importing torch or fitting models.

Requires numpy and matplotlib. Reads the original integrated reports to recover
effective architecture dimensions and the authors' binary, PA-adjusted AUC.
Combined plots are exploratory: scoring implementation parity is not certified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'toast-mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATASETS = ('OPS-SAT', 'SMAP', 'MSL')
SEEDS = (2026, 2027, 2028)
BASELINES = {'OPS-SAT': (10, 8), 'SMAP': (10, 50), 'MSL': (50, 110)}
PAPER = (
    ('pot_f1', 'POT F1 · con PA'),
    ('dense_f1', 'Best F1 · griglia densa · con PA'),
    ('pot_auc', 'POT ROC/AUC · binaria dopo PA'),
    ('dense_auc', 'Best-F1 denso ROC/AUC · binaria dopo PA'),
)
PRECISION_RECALL = (
    ('pot_precision', 'POT · precision'), ('pot_recall', 'POT · recall'),
    ('dense_precision', 'Best F1 denso · precision'), ('dense_recall', 'Best F1 denso · recall'),
)
RAW = (
    ('raw_f1', 'F1 raw · soglia dal training'),
    ('raw_auc', 'ROC-AUC · score continui'),
    ('raw_ap', 'Average precision · score continui'),
)
FOOTER = ('Griglia esplorativa v10 + v12 · revisioni del codice di valutazione diverse: '
          'equivalenza numerica da verificare.')


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite metric: {value}')
    return value


def recover_dense_metrics(bf):
    """Recover integer PA confusion counts from saved dense precision/recall.

    F1 alone is insufficient. Class totals are invariant under threshold/alpha
    and are available in the original-BF confusion matrix for the same run.
    Reject ambiguous or inconsistent reconstructions instead of estimating AUC.
    """
    positives = number(bf['bf_TP']) + number(bf['bf_FN'])
    negatives = number(bf['bf_TN']) + number(bf['bf_FP'])
    precision = number(bf['bf_dense_precision'])
    recall = number(bf['bf_dense_recall'])
    if positives <= 0 or negatives <= 0 or precision <= 0 or recall <= 0:
        raise ValueError('Dense AUC cannot be recovered uniquely from these saved metrics')
    tp_float = recall * positives
    tp = round(tp_float)
    fp_float = tp * (1 / precision - 1)
    fp = round(fp_float)
    if (abs(tp_float-tp) > 1e-6 or abs(fp_float-fp) > 1e-6
            or not 0 <= tp <= positives or not 0 <= fp <= negatives):
        raise ValueError('Dense metrics do not determine consistent integer confusion counts')
    tn, fn = round(negatives)-fp, round(positives)-tp
    f1 = 2*tp / (2*tp+fp+fn)
    for actual, expected in ((tp/(tp+fp), precision), (tp/(tp+fn), recall),
                             (f1, number(bf['bf_dense_f1']))):
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Recovered dense confusion counts do not reproduce saved metrics')
    return {'dense_f1': f1, 'dense_precision': precision, 'dense_recall': recall,
            'dense_tp': tp, 'dense_tn': tn, 'dense_fp': fp, 'dense_fn': fn,
            'dense_auc': .5 * (tp/(tp+fn) + tn/(tn+fp))}


def load_reports(results):
    rows, provenance, split_signatures = [], {}, set()
    for version, expected in (('v10', 144), ('v12', 126)):
        root = results / version / 'study'
        csvpath = root / 'all_experiments.csv'
        entries = list(csv.DictReader(csvpath.open(encoding='utf-8')))
        if len(entries) != expected:
            raise ValueError(f'{version}: expected {expected} runs, found {len(entries)}')
        provenance[version] = {
            'csv': str(csvpath), 'csv_sha256': hashlib.sha256(csvpath.read_bytes()).hexdigest(),
            'source_hashes': {},
        }
        for entry in entries:
            report_path = (root / entry['report_file']).resolve()
            if not report_path.is_relative_to(root.resolve()):
                raise ValueError(f'Report outside result directory: {report_path}')
            report = json.loads(report_path.read_text(encoding='utf-8'))
            hp = report['profile']
            if report['identifier'] != entry['identifier'] or report['dataset'] != entry['dataset']:
                raise ValueError(f'Identity mismatch: {report_path}')
            if report['checkpoint']['sha256'] != entry['checkpoint_sha256']:
                raise ValueError(f'Checkpoint mismatch: {report_path}')
            seed = int(entry['seed'])
            if seed != int(hp['seed']) or seed not in SEEDS or int(hp['epochs']) != 50:
                raise ValueError(f'Seed or epoch mismatch: {report_path}')
            if not report['measurement_settings']['paper_alpha_search']:
                raise ValueError(f'Expected paper alpha search: {report_path}')
            settings = report['measurement_settings']
            if settings['alpha'] != .5 or settings['threshold_train_quantile'] != .995:
                raise ValueError(f'Unexpected raw-score settings: {report_path}')
            for key, digest in report['source'].items():
                if key.endswith('sha256'):
                    provenance[version]['source_hashes'].setdefault(key, set()).add(digest)
            paper = report['original_paper_evaluation']['metrics']
            row = dict(campaign=version, dataset=entry['dataset'], identifier=entry['identifier'],
                       window=int(hp['window_sz']), feedforward=int(hp['dim_feedforward']),
                       seed=seed, epochs=50, report_file=str(report_path))
            for prefix in ('pot', 'bf'):
                metrics = paper[prefix]
                row[prefix + '_f1'] = number(metrics[prefix + '_f1'])
                row[prefix + '_auc'] = number(metrics[prefix + '_ROC/AUC'])
                for key in ('precision', 'recall'):
                    row[prefix + '_' + key] = number(metrics[prefix + '_' + key])
                tp, tn, fp, fn = (number(metrics[prefix + '_' + key]) for key in ('TP', 'TN', 'FP', 'FN'))
                expected_auc = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
                if not math.isclose(row[prefix + '_auc'], expected_auc, abs_tol=1e-10):
                    raise ValueError(f'Binary AUC definition mismatch: {report_path}')
            row.update(recover_dense_metrics(paper['bf']))
            for target, source in (
                ('raw_f1', 'f1_raw'), ('raw_auc', 'roc_auc_raw'), ('raw_ap', 'average_precision_raw'),
                ('parameters', 'parameters'), ('macs', 'estimated_macs_per_window'),
                ('latency_ms', 'latency_p50_ms'),
            ):
                row[target] = number(entry[source])
            raw = report['raw_pointwise_evaluation']
            cm = raw['confusion_matrix']
            row['point_prevalence'] = (number(cm['tp']) + number(cm['fn'])) / sum(map(number, cm.values()))
            if row['dataset'] == 'OPS-SAT':
                inv = report['dataset_inventory']
                signature = (inv['source_sha256'], *(tuple(inv[k]) for k in (
                    'train_fragment_ids', 'validation_fragment_ids', 'test_fragment_ids')))
                split_signatures.add(signature)
                selected = report['validation_selected_evaluation']
                if selected['status'] != 'measured':
                    raise ValueError(f'Missing validation-selected OPS-SAT results: {report_path}')
                row['validation_auc'] = number(selected['selection']['chosen']['validation_fragment_auc'])
                row['selected_test_auc'] = number(selected['fragment']['roc_auc'])
                row['selected_test_ap'] = number(selected['fragment']['average_precision'])
                row['fragment_prevalence'] = (selected['fragment']['n_anomalous_fragments'] /
                                              selected['fragment']['n_fragments'])
            rows.append(row)
    if len({r['identifier'] for r in rows}) != 270 or len(split_signatures) != 1:
        raise ValueError('Duplicate IDs or different OPS-SAT source/split inventories')
    groups = {}
    for row in rows:
        key = row['dataset'], row['window'], row['feedforward']
        groups.setdefault(key, []).append(row)
    cells = []
    for (dataset, window, ff), group in sorted(groups.items()):
        if sorted(r['seed'] for r in group) != list(SEEDS) or len({r['campaign'] for r in group}) != 1:
            raise ValueError(f'Incomplete or duplicated architecture: {dataset, window, ff}')
        cell = dict(dataset=dataset, window=window, feedforward=ff, campaign=group[0]['campaign'], n=3)
        for key, value in group[0].items():
            if isinstance(value, (int, float)) and key not in ('window', 'feedforward', 'seed', 'epochs'):
                values = [r[key] for r in group]
                cell[key + '_mean'] = statistics.mean(values)
                cell[key + '_std'] = statistics.stdev(values)
        cells.append(cell)
    for dataset in DATASETS:
        part = [c for c in cells if c['dataset'] == dataset]
        if len(part) != 30 or len({c['window'] for c in part}) != 6 or len({c['feedforward'] for c in part}) != 5:
            raise ValueError(f'Incomplete 6 x 5 grid: {dataset}')
    for item in provenance.values():
        item['source_hashes'] = {k: sorted(v) for k, v in item['source_hashes'].items()}
    model_hashes = {h for v in ('v10', 'v12') for h in provenance[v]['source_hashes']['models_sha256']}
    if len(model_hashes) != 1:
        raise ValueError('Model code differs between runs')
    provenance['notes'] = [
        '270 reports checked; 90 architecture cells; three seeds per cell; 50 epochs.',
        'Authors ROC/AUC verified against balanced accuracy of binary PA predictions.',
        'Dense ROC/AUC recovered for all 270 runs from integer confusion counts, reconstructed '
        'using saved dense precision/recall and label totals. Saved dense F1 verified to 1e-12.',
        'OPS-SAT source hash and exact train/validation/test segment IDs match across all runs.',
        'Model source hashes match; evaluation source hashes differ. Numeric scoring parity is NOT certified.',
        'No retraining or selection of a final architecture was performed.',
    ]
    return rows, cells, provenance


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def grid(cells, dataset):
    part = [c for c in cells if c['dataset'] == dataset]
    windows = sorted({c['window'] for c in part})
    ffs = sorted({c['feedforward'] for c in part})
    lookup = {(c['window'], c['feedforward']): c for c in part}
    return windows, ffs, lookup


def heatmap(fig, ax, windows, ffs, values, title, *, baseline=None, fmt='.3f', cmap='viridis', limits=None):
    values = np.asarray(values, dtype=float)
    if limits is None:
        lo, hi = float(values.min()), float(values.max())
        pad = max((hi - lo) * .06, .0005)
        limits = lo - pad, hi + pad
    im = ax.imshow(values, origin='lower', aspect='auto', cmap=cmap, vmin=limits[0], vmax=limits[1])
    ax.set(xticks=range(len(windows)), xticklabels=windows, yticks=range(len(ffs)), yticklabels=ffs,
           xlabel='Window size W (campioni)', ylabel='Dimensione feed-forward', title=title)
    for j in range(len(ffs)):
        for i in range(len(windows)):
            rgba = im.cmap(im.norm(values[j, i]))
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            ax.text(i, j, format(values[j, i], fmt), ha='center', va='center',
                    color='#111827' if luminance > .53 else 'white', fontsize=9)
    if baseline is not None:
        i, j = windows.index(baseline[0]), ffs.index(baseline[1])
        ax.add_patch(Rectangle((i-.47, j-.47), .94, .94, fill=False, lw=2.2, edgecolor='#ef4444'))
    fig.colorbar(im, ax=ax, fraction=.046, pad=.035).ax.tick_params(labelsize=8)


class Figures:
    def __init__(self, output):
        self.output = output
        self.directory = output / 'figures'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.book = PdfPages(output / 'ablation_v10_v12.pdf')
        self.items = []

    def save(self, fig, name, title, subtitle):
        fig.suptitle(title + '\n' + subtitle, fontsize=14, y=.995)
        fig.text(.5, .012, FOOTER, ha='center', fontsize=8, color='#52525b')
        fig.tight_layout(rect=(0, .04, 1, .94))
        fig.savefig(self.directory / (name + '.png'), dpi=170, facecolor='white')
        fig.savefig(self.directory / (name + '.pdf'), facecolor='white')
        self.book.savefig(fig, facecolor='white')
        self.items.append((name, title, subtitle))
        plt.close(fig)


def make_figures(cells, out):
    figs = Figures(out)
    try:
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
        for ax, dataset in zip(axs, DATASETS):
            w, ff, lookup = grid(cells, dataset)
            values = [[0 if lookup[a, b]['campaign'] == 'v10' else 1 for a in w] for b in ff]
            heatmap(fig, ax, w, ff, values, dataset, baseline=BASELINES[dataset], fmt='.0f',
                    cmap=ListedColormap(['#cbd5e1', '#60a5fa']), limits=(-.5, 1.5))
            ax.images[0].colorbar.set_ticks([0, 1], labels=['v10', 'v12'])
        figs.save(fig, '01_disegno_griglia', 'Disegno sperimentale e provenienza delle configurazioni',
                  '0 = v10 · 1 = v12 · bordo rosso = baseline · ogni cella contiene 3 seed e 50 epoche')

        for dataset in DATASETS:
            tag = dataset.lower().replace('-', '')
            w, ff, lookup = grid(cells, dataset)
            fig, axs = plt.subplots(2, 2, figsize=(14, 10.5))
            for ax, (key, label) in zip(axs.flat, PAPER):
                values = [[lookup[a, b][key + '_mean'] for a in w] for b in ff]
                heatmap(fig, ax, w, ff, values, label, baseline=BASELINES[dataset])
            figs.save(fig, f'02_{tag}_metriche_paper', f'{dataset} · point adjustment e Best F1 denso',
                      'Medie su 3 seed · ROC/AUC binaria come nel codice degli autori · scale colore locali · bordo rosso = baseline')

            fig, axs = plt.subplots(2, 2, figsize=(14, 10.5))
            for ax, (key, label) in zip(axs.flat, PRECISION_RECALL):
                values = [[lookup[a, b][key + '_mean'] for a in w] for b in ff]
                heatmap(fig, ax, w, ff, values, label, baseline=BASELINES[dataset])
            figs.save(fig, f'02_{tag}_precision_recall', f'{dataset} · precision e recall con PA e Best F1 denso',
                      'Medie su 3 seed · predizioni dopo PA · α / soglia scelti per F1, non per precision o recall · scale locali')

            fig, axs = plt.subplots(4, 2, figsize=(14, 16))
            for row, (key, label) in enumerate(PAPER):
                for col, (xs, series, flip, series_name) in enumerate(((w, ff, False, 'FF'), (ff, w, True, 'W'))):
                    ax = axs[row, col]
                    colors = plt.cm.viridis(np.linspace(.08, .88, len(series)))
                    for level, color in zip(series, colors):
                        cc = [lookup[level, x] if flip else lookup[x, level] for x in xs]
                        means = np.array([c[key + '_mean'] for c in cc])
                        sd = np.array([c[key + '_std'] for c in cc])
                        ax.plot(xs, means, color=color, lw=1.5, label=f'{series_name}={level}')
                        ax.fill_between(xs, means-sd, means+sd, color=color, alpha=.09)
                        for x, c, y, err in zip(xs, cc, means, sd):
                            ax.errorbar(x, y, yerr=err, color=color, marker='s' if c['campaign']=='v10' else 'o',
                                        markersize=4.2, capsize=2, lw=.65)
                    ax.set(title=label, xlabel='Dimensione feed-forward' if flip else 'Window size W (campioni)',
                           ylabel='Media ± 1 deviazione standard', xticks=xs)
                    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]), top=min(1.01, ax.get_ylim()[1]))
                    ax.grid(alpha=.2)
                    if row == 0:
                        ax.legend(fontsize=8, ncols=3, loc='best')
            figs.save(fig, f'03_{tag}_trend_paper', f'{dataset} · trend al variare di W e FF',
                      'Tre seed · bande e barre = ±1 SD (non intervalli di confidenza) · quadrati v10, cerchi v12 · linee guida visive')

            fig, axs = plt.subplots(1, 3, figsize=(17, 5))
            for ax, (key, label) in zip(axs, RAW):
                values = [[lookup[a, b][key + '_mean'] for a in w] for b in ff]
                heatmap(fig, ax, w, ff, values, label, baseline=BASELINES[dataset])
            figs.save(fig, f'04_{tag}_metriche_raw', f'{dataset} · valutazione puntuale senza point adjustment',
                      'Medie su 3 seed · α = 0,5 · soglia = quantile 99,5% del training · scale colore locali')

        w, ff, lookup = grid(cells, 'OPS-SAT')
        fig, axs = plt.subplots(1, 3, figsize=(17, 5))
        for ax, (key, label) in zip(axs, (
            ('validation_auc', 'Validation · ROC-AUC per segmento'),
            ('selected_test_auc', 'Test · ROC-AUC per segmento'),
            ('selected_test_ap', 'Test · average precision per segmento'),
        )):
            heatmap(fig, ax, w, ff, [[lookup[a, b][key + '_mean'] for a in w] for b in ff],
                    label, baseline=BASELINES['OPS-SAT'])
        figs.save(fig, '05_opssat_validation_segmenti', 'OPS-SAT · scoring scelto sulla validation',
                  'Medie su 3 seed · α / calibrazione / aggregazione selezionati su validation · nessun PA · scale locali')

        fig, axs = plt.subplots(3, 3, figsize=(17, 13))
        for row, dataset in enumerate(DATASETS):
            w, ff, lookup = grid(cells, dataset)
            base = lookup[BASELINES[dataset]]
            for ax, (key, label) in zip(axs[row], (
                ('parameters', 'Parametri appresi (% baseline)'), ('macs', 'MAC stimati (% baseline)'),
                ('latency_ms', 'Latenza p50 A100 (ms)'),
            )):
                values = [[lookup[a, b][key + '_mean'] * (100 / base[key + '_mean'] if key!='latency_ms' else 1)
                           for a in w] for b in ff]
                heatmap(fig, ax, w, ff, values, dataset + ' · ' + label,
                        baseline=BASELINES[dataset], fmt='.1f' if key!='latency_ms' else '.2f', cmap='magma_r')
        figs.save(fig, '06_risorse', 'Costo del modello e tempi misurati',
                  'Parametri / MAC rispetto alla baseline del dataset · latenza: media delle p50 sulle seed · A100, non hardware di bordo')

        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        for col, dataset in enumerate(DATASETS):
            w, ff, lookup = grid(cells, dataset)
            base = lookup[BASELINES[dataset]]
            for row, (metric, label) in enumerate(PAPER[:2]):
                ax = axs[row, col]
                for c in lookup.values():
                    x = 100*c['macs_mean']/base['macs_mean']
                    y, sd = c[metric+'_mean'], c[metric+'_std']
                    color = plt.cm.viridis(.08 + .8*w.index(c['window'])/(len(w)-1))
                    ax.errorbar(x, y, yerr=sd, color=color, lw=.6, alpha=.65, capsize=2)
                    ax.scatter(x, y, color=color, s=30+70*c['feedforward']/max(ff),
                               marker='s' if c['campaign']=='v10' else 'o', edgecolors='white', linewidth=.4)
                ax.scatter(100, base[metric+'_mean'], marker='*', s=210, color='#ef4444', zorder=5)
                ax.set(title=dataset+' · '+label, xlabel='MAC stimati (% baseline)', ylabel='Media ± 1 SD')
                ax.grid(alpha=.2)
                ax.set_ylim(bottom=max(0, ax.get_ylim()[0]), top=min(1.01, ax.get_ylim()[1]))
                if row == 0:
                    handles = [Line2D([], [], color=plt.cm.viridis(.08+.8*i/(len(w)-1)), marker='o',
                                      linestyle='', label=f'W={level}') for i, level in enumerate(w)]
                    ax.legend(handles=handles, ncols=3, fontsize=8, loc='best')
        figs.save(fig, '07_prestazioni_vs_calcolo', 'Prestazioni con PA e Best F1 denso rispetto al costo computazionale',
                  'Stella rossa = baseline · colore = W · dimensione punto = FF · quadrati v10, cerchi v12 · nessuna selezione finale sul test')

        fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
        for ax, dataset in zip(axs, DATASETS):
            part = [c for c in cells if c['dataset']==dataset]
            for c in part:
                ax.errorbar(c['bf_f1_mean'], c['dense_f1_mean'], xerr=c['bf_f1_std'], yerr=c['dense_f1_std'],
                            fmt='s' if c['campaign']=='v10' else 'o', markersize=4, lw=.65, alpha=.65,
                            color='#475569' if c['campaign']=='v10' else '#2563eb')
            lower = max(0, min(c['bf_f1_mean']-c['bf_f1_std'] for c in part)-.03)
            upper = min(1.01, max(c['dense_f1_mean']+c['dense_f1_std'] for c in part)+.03)
            ax.plot([lower, upper], [lower, upper], '--', color='#ef4444', lw=1)
            ax.set(xlim=(lower, upper), ylim=(lower, upper), title=dataset,
                   xlabel='Best F1 · 200 soglie · con PA', ylabel='Best F1 · ricerca densa · con PA')
            ax.grid(alpha=.2)
        figs.save(fig, '08_diagnostica_soglie', 'Sensibilità alla ricerca della soglia',
                  'Media ± 1 SD · entrambe le metriche ottimizzate sul test (oracle) · diagonale = uguaglianza · v10 grigio, v12 blu')
    finally:
        figs.book.close()
    return figs.items


def readme(output, items):
    text = '''# Ablation TranAD+ — v10 e v12

## Aprire i grafici

- [PDF completo, con tutti i grafici](ablation_v10_v12.pdf).
- Ogni figura è disponibile anche in PNG e PDF nella cartella `figures/`.
- [270 run con metriche e dimensioni effettive](runs_270.csv).
- [90 architetture: medie e deviazioni standard](architecture_mean_std.csv).
- [Provenienza e controlli sui dati](provenance.json).

## Come leggere le figure

Ogni cella rappresenta tre seed (2026, 2027, 2028), con 50 epoche di training.
La griglia combinata contiene 30 architetture per dataset: 6 finestre × 5 larghezze FF.
W e FF sono dimensioni effettive, non le percentuali presenti negli identificatori v10.
Il bordo rosso indica la baseline del dataset. Le scale colore delle heatmap sono
locali al pannello: confrontare i numeri e la colorbar, non il colore fra pannelli.
Le barre mostrano una deviazione standard fra seed, non intervalli di confidenza.
Le linee dei trend collegano le osservazioni e non sono modelli interpolanti.

## Definizioni delle metriche

- **Metriche paper**: POT / Best F1 con point adjustment. Alpha è scelto sul test
  secondo il rispettivo F1; Best F1 sceglie anche la soglia sul test. Sono risultati oracle.
  I grafici principali usano **Best F1 denso**, l'estensione esaustiva della griglia
  originale; il Best F1 delle 200 soglie resta nel confronto diagnostico finale.
- **ROC/AUC paper**: `roc_auc_score` sulle predizioni BINARIE dopo PA, equivalente
  alla balanced accuracy. Non è la ROC-AUC degli anomaly score continui.
- **Raw**: score continui con alpha=0.5; F1 alla soglia del quantile 99.5% del training;
  nessun PA. ROC-AUC e average precision usano gli score continui, non le predizioni binarie.
- **OPS-SAT validation**: alpha, calibrazione e aggregazione scelti sulla validation
  separatamente per ogni run. La ROC-AUC/average precision test è a livello di segmento.
  I punti non hanno etichette precise di onset/offset. Le metriche a livello di segmento
  e quelle puntuali rispondono a domande diverse.
- **Average precision** è AP, non l'integrale trapezoidale della curva precision-recall.
- **Ricerca densa**: ricerca oracle su tutti gli score-soglia distinti, ancora con PA.
  F1, precision, recall e ROC/AUC dense si riferiscono allo STESSO alpha e alla STESSA
  soglia scelti per massimizzare l'F1 denso. La ROC/AUC non è massimizzata separatamente.
  Nei CSV `dense_*` identifica questa soluzione; `bf_*` conserva la soluzione a 200 soglie.
- **ROC/AUC densa ricostruita**: dato il numero di positivi P e negativi N del test,
  TP = recall_densa × P; FP = TP × (1 / precision_densa - 1); FN = P - TP; TN = N - FP.
  La ROC/AUC binaria è (TP/P + TN/N) / 2. Sono verificati l'integralità dei conteggi e
  l'accordo con precision, recall e F1 salvati (tolleranza assoluta 1e-12). I 270 report
  contengono precision/recall positive: nessuna ricostruzione ambigua è stata utilizzata.
  Il solo F1, senza precision/recall e conteggi di classe, non sarebbe sufficiente.
- **MAC**: stima del calcolo per finestra, non misura di energia o latenza a bordo.
  Le latenze sono forward PyTorch su A100, espresse come media delle p50 delle tre seed.

## Provenienza e limiti

I CSV e i 270 report sono stati letti senza modificare i risultati originali.
Sono controllati: identità, checksum dichiarato del checkpoint, completezza delle
seed, dimensioni della griglia, epoche e definizione della ROC/AUC binaria.
I segmenti train/validation/test e l'hash del dataset OPS-SAT coincidono fra tutti i run.
Le revisioni del codice di valutazione v10/v12 differiscono. Le figure combinate sono
esplorative: non certificano l'equivalenza numerica delle due implementazioni.
La mappa iniziale permette di identificare le celle di ciascuna campagna.
Le differenze fra seed non esauriscono l'incertezza: lo split e il test sono gli stessi.
Nessuna architettura finale è selezionata da questi grafici del test.

## Riproduzione

Da questa repository:

```bash
python3 tranad_plus/reports/plot_ablation_grid.py
```

Dipendenze: numpy e matplotlib. Non serve PyTorch; non si avviano training o valutazioni.

## Galleria

'''
    for name, title, subtitle in items:
        text += f'### {title}\n\n{subtitle}\n\n[PNG](figures/{name}.png) · [PDF](figures/{name}.pdf)\n\n'
        text += f'![{title}](figures/{name}.png)\n\n'
    (output/'README.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=PROJECT/'results')
    parser.add_argument('--output', type=Path, default=PROJECT/'results'/'analysis_v10_v12')
    args = parser.parse_args()
    rows, cells, provenance = load_reports(args.results_root.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'runs_270.csv', rows)
    write_csv(output/'architecture_mean_std.csv', cells)
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titleweight': 'semibold',
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    items = make_figures(cells, output)
    readme(output, items)
    print(f'Validated {len(rows)} runs / {len(cells)} architecture cells; generated {len(items)} figures.')
    print(output/'ablation_v10_v12.pdf')
    print(output/'README.md')


if __name__ == '__main__':
    main()
