#!/usr/bin/env python3
"""Informal results discussion: BF200/POT, matched PA/raw controls, real traces."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import sys

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/toast-matplotlib')
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREVIEW = ROOT / 'presentations/2026-11-11_ablation_preview'
sys.path.insert(0, '/tmp/toast-presentation-deps')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from pptx import Presentation
from pptx.util import Inches

spec = importlib.util.spec_from_file_location('original_deck', ROOT / 'presentations/2026-11-03_progress/build.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
RED, INK, GRAY = old.RED, old.INK, old.GRAY
DS = ('OPS-SAT', 'SMAP', 'MSL')
FIG = OUT / 'figures'
DATA = OUT / 'data'
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 17,
                     'axes.titlesize': 21, 'axes.labelsize': 17,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14})

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def readcsv(path):
    with path.open() as f:
        return list(csv.DictReader(f))

cells = readcsv(PREVIEW / 'cell_summary.csv')
for c in cells:
    for k in ('window', 'feedforward', 'seeds', 'epochs'):
        c[k] = int(c[k])
    for k in list(c):
        if k.endswith(('_mean', '_sd')):
            c[k] = float(c[k])
assert len(cells) == 90
lookup = {(c['dataset'], c['window'], c['feedforward']): c for c in cells}
grid = {ds: (sorted({c['window'] for c in cells if c['dataset'] == ds}),
             sorted({c['feedforward'] for c in cells if c['dataset'] == ds})) for ds in DS}
sources = json.loads((PREVIEW / 'provenance.json').read_text())
assert sha(ROOT / sources['source_csv']) == sources['source_sha256']
for report in sources['reports']:
    assert sha(ROOT / report['report']) == report['sha256']
# Recompute each displayed mean/SD from the report-verified source rows.
runs = readcsv(PREVIEW / 'source_runs.csv')
for c in cells:
    selected = [r for r in runs if r['dataset'] == c['dataset'] and int(r['window']) == c['window']
                and int(r['feedforward']) == c['feedforward']]
    assert len(selected) == 3
    for method in ('bf', 'pot'):
        for metric in ('f1', 'precision', 'recall', 'auc'):
            values = np.array([100 * float(r[method + '_' + metric]) for r in selected])
            np.testing.assert_allclose(values.mean(), c[method + '_' + metric + '_mean'], rtol=1e-12)
            np.testing.assert_allclose(values.std(ddof=1), c[method + '_' + metric + '_sd'], atol=1e-12)

evidence_path = ROOT / 'tesi/data/chapter4/benchmark-evidence.json'
evidence = json.loads(evidence_path.read_text())
bench = {}
for ds in old.ORDER:
    for method in ('pot', 'bf'):
        # WADI POT is explicitly the masked diagnostic, not an ordinary maximum.
        row = next(r for r in evidence['benchmark_rows'] if r['dataset'] == ds and r['method'] == method)
        source = ROOT / row['source']
        assert sha(source) == row['sha256']
        result = dict(row)
        if ds == 'MSL' and method == 'pot':
            path = ROOT / 'results/legacy/MSL/_native_pickles/papercheck_MSL_pot_w10_ff110_l3_bs8192_s1_pot.pkl'
            values = pickle.load(path.open('rb'))
            result = {'dataset': ds, 'method': method, 'source': str(path.relative_to(ROOT)),
                      'sha256': sha(path), 'metrics': {'f1': float(values['pot_f1'])}}
        if source.suffix == '.pkl' and not (ds == 'MSL' and method == 'pot'):
            values = pickle.load(source.open('rb'))
            assert float(values[method + '_f1']) == row['metrics']['f1']
        if ds == 'WADI' and method == 'pot':
            assert row['metrics']['f1'] == evidence['wadi_diagnostics']['local_masked']['f1']
            mask = evidence['wadi_diagnostics']['mask']
            assert sha(ROOT / mask['source']) == mask['sha256']
        bench[ds, method] = result

def canvas(heading):
    fig = old.canvas()
    fig.text(.055, .95, heading, color=RED, fontsize=27, va='top')
    return fig

def footer(fig, text):
    fig.text(.055, .045, text, fontsize=12, color=GRAY, va='bottom')

def benchmark():
    fig = canvas('Benchmark reproduction · F1 (%)')
    ax = old.plain_axis(fig)
    xs = [.075, .315, .465, .690, .840]
    ax.text(.420, .825, 'POT', ha='center', color=RED, fontsize=23)
    ax.text(.790, .825, 'Best F1 · 200 thresholds', ha='center', color=RED, fontsize=21)
    for x, text in zip(xs, ['Dataset', 'Paper', 'Ours', 'Paper', 'Ours']):
        ax.text(x, .748, text, fontsize=20, weight='bold' if x == xs[0] else 'normal')
    ax.plot([.065, .94], [.72, .72], color=RED, lw=1.5)
    for i, ds in enumerate(old.ORDER):
        y = .662 - i * .078
        if i % 2 == 0:
            ax.add_patch(Rectangle((.065, y-.021), .875, .065, color='#FAF5F6', linewidth=0))
        paper = evidence['paper_source']['values'][ds]
        values = [ds, f"{100*paper['pot'][0]:.2f}", f"{100*bench[ds,'pot']['metrics']['f1']:.2f}" + ('*' if ds == 'WADI' else ''),
                  f"{100*paper['bf'][0]:.2f}", f"{100*bench[ds,'bf']['metrics']['f1']:.2f}"]
        for x, value in zip(xs, values):
            ax.text(x, y, value, fontsize=22, va='center', color=INK)
    ax.text(.075, .145, '* WADI POT: NaN masking applied.', fontsize=14, color=GRAY)
    footer(fig, 'Point adjustment · best available individual runs, selected separately for each method')
    return fig

def trends(direction):
    fig = canvas('Window size' if direction == 'window' else 'Feed-forward dimension')
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=.065, right=.975, bottom=.20, top=.70, wspace=.23)
    for ax, ds in zip(axes, DS):
        ws, hs = grid[ds]
        xs = ws if direction == 'window' else hs
        cs = [lookup[ds, x, max(hs)] if direction == 'window' else lookup[ds, max(ws), x] for x in xs]
        for method, color, label, marker in [('bf', RED, 'Best F1 · 200 thresholds', 'o'), ('pot', '#5C5C5C', 'POT', 's')]:
            means = np.array([c[method+'_f1_mean'] for c in cs])
            sd = np.array([c[method+'_f1_sd'] for c in cs])
            ax.errorbar(xs, means, yerr=sd, color=color, marker=marker, lw=2, ms=5, capsize=3, label=label)
            ax.fill_between(xs, means-sd, means+sd, color=color, alpha=.07, linewidth=0)
        ax.set_title(ds, loc='left', pad=32)
        ax.text(0, 1.035, f'FF = {max(hs)}' if direction == 'window' else f'W = {max(ws)}',
                transform=ax.transAxes, color=GRAY, fontsize=15)
        ax.set_xticks(xs)
        # Preserve small within-dataset effects; shared normalization is for heatmaps.
        ax.set_ylim(*{'OPS-SAT': (40, 100), 'SMAP': (70, 90), 'MSL': (15, 100)}[ds])
        ax.set_xlabel('W (samples)' if direction == 'window' else 'FF')
        ax.set_ylabel('F1 (%)' if ds == DS[0] else '')
        ax.grid(axis='y', color='#E6E6E6', lw=.8)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.55, .865), fontsize=16)
    footer(fig, 'Mean ± sample SD · 3 seeds · 50 epochs · PA · dataset-specific vertical scales')
    return fig

# Exact palette and vector-cell rendering used in the first presentation.
cmap = LinearSegmentedColormap.from_list('Padova', ['#FFF9FA','#F7DDE2','#DA8F9D',RED])
heatmap_limits = {}
def heatmap(method):
    title = 'Best F1 · 200 thresholds' if method == 'bf' else 'POT'
    fig = canvas('Window × feed-forward dimension · '+title)
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=.07, right=.975, bottom=.30, top=.79, wspace=.30)
    heatmap_limits[method] = {}
    for ax, ds in zip(axes, DS):
        ws, hs = grid[ds]
        values = np.array([[lookup[ds, w, h][method+'_f1_mean'] for w in ws] for h in hs])
        lo, hi = np.floor(values.min()), np.ceil(values.max())
        hi = max(hi, lo+1)
        heatmap_limits[method][ds] = [float(lo), float(hi)]
        norm = Normalize(lo, hi)
        im = ax.pcolormesh(np.arange(len(ws)+1)-.5,np.arange(len(hs)+1)-.5,
                          values,cmap=cmap,norm=norm,shading='flat',rasterized=False,
                          edgecolors='white',linewidth=.7)
        ax.set_title(ds, loc='left', pad=14)
        ax.set_xticks(range(len(ws)), ws)
        ax.set_yticks(range(len(hs)), hs)
        ax.set_xlabel('W (samples)', labelpad=8)
        ax.set_ylabel('FF', labelpad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(ws)), minor=True)
        ax.set_yticks(np.arange(-.5, len(hs)), minor=True)
        ax.grid(which='minor', color='white', lw=1.3)
        ax.tick_params(which='both', length=0)
        for j in range(len(hs)):
            for i in range(len(ws)):
                rgb = cmap(norm(values[j, i]))
                lum = .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
                ax.text(i, j, f'{values[j,i]:.1f}', fontsize=15, ha='center', va='center',
                        color='white' if lum < .49 else INK)
        bounds = ax.get_position()
        cax = fig.add_axes([bounds.x0, .14, bounds.width, .022])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.outline.set_visible(False)
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor('face')
        cb.set_label('Mean F1 (%)', fontsize=14)
        cb.ax.tick_params(labelsize=11)
    footer(fig, '3 seeds per cell · 50 epochs · PA · separate colour scale for each dataset')
    return fig

signal_sources = []
native_summary = []
real_examples = {}
example_candidates = []
for ds in ('SMAP', 'MSL'):
    base = ROOT / 'data/processed/tranad_plus' / ds
    lows, highs = [], []
    for path in sorted((base / 'Train/Tracks').glob('*.npy')):
        x = np.load(path, mmap_mode='r')[:, 0]
        lows.append(float(x.min())); highs.append(float(x.max()))
        signal_sources.append({'file':str(path.relative_to(ROOT)), 'sha256':sha(path)})
    span = max(highs)-min(lows)
    steps, labels, events = [], [], []
    for path in sorted((base / 'Test/Tracks').glob('*.npy')):
        x = np.load(path, mmap_mode='r')[:, 0]
        labelpath = base / 'Test/Labels' / path.name
        y = np.load(labelpath).sum(axis=1) > 0
        # No boundary difference and no synthetic zero at the first sample.
        steps.append(np.abs(np.diff(x))/span)
        labels.append(y[1:])
        edges = np.diff(np.r_[False,y,False].astype(int))
        for start,stop in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if stop-start < 20:
                continue
            d = np.abs(np.diff(x[start:stop]))/span
            events.append({'dataset':ds,'track':path.name,'start':int(start),'stop':int(stop),
                           'length':int(stop-start),'flat_fraction':float(np.mean(d==0)),
                           'q95_step':float(np.quantile(d,.95))})
        signal_sources.extend({'file':str(p.relative_to(ROOT)), 'sha256':sha(p)} for p in (path,labelpath))
    steps, labels = np.concatenate(steps), np.concatenate(labels)
    for label in (False, True):
        v = steps[labels == label]
        native_summary.append({'dataset':ds, 'label':int(label), 'within_track_steps':len(v),
                               'zero_step_fraction':float(np.mean(v == 0)),
                               'q95_abs_step_divided_by_train_range':float(np.quantile(v,.95))})
    nominal_q95 = float(np.quantile(steps[~labels],.95))
    for event in events:
        event['eligible'] = (event['flat_fraction']>=.5 if ds=='SMAP' else event['q95_step']>nominal_q95)
    eligible = sorted([r for r in events if r['eligible']],key=lambda r:(r['length'],r['track'],r['start']))
    chosen = dict(eligible[(len(eligible)-1)//2])
    chosen['eligible_events'] = len(eligible)
    chosen['selection'] = ('Median-length event among positive runs ≥20 samples with ≥50% identical steps'
                           if ds=='SMAP' else 'Median-length event among positive runs ≥20 samples with step q95 above pooled nominal q95')
    path = base/'Test/Tracks'/chosen['track']
    x = np.load(path)[:,0]
    y = np.load(base/'Test/Labels'/path.name).sum(axis=1)>0
    a,b = max(0,chosen['start']-80),min(len(x),chosen['stop']+80)
    chosen['display_start'],chosen['display_stop'] = a,b
    chosen['source'] = str(path.relative_to(ROOT))
    chosen['source_sha256'] = sha(path)
    chosen['indices'] = np.arange(a,b)
    chosen['values'] = x[a:b]
    chosen['labels'] = y[a:b]
    real_examples[ds] = chosen
    old.csv_write(DATA/(ds.lower()+'_real_example.csv'),[
        {'sample_index':int(t),'telemetry_coordinate':float(v),'label':int(label)}
        for t,v,label in zip(chosen['indices'],chosen['values'],chosen['labels'])])
    example_candidates.extend(events)
old.csv_write(DATA / 'native_signal_summary.csv', native_summary)
old.csv_write(DATA / 'example_candidates.csv', example_candidates)

# Read matched FP32 reference-model controls; no additional inference or training.
no_pa_runs, no_pa_sources = [], []
for ds in DS:
    for config in ('reference','window_only'):
        for seed in (2026,2027,2028):
            folder = ROOT/f'results/v18/no_pa/{ds}/reference/{config}/seed_{seed}/fp32'
            path = folder/'metrics.json'
            status = json.loads((folder/'status.json').read_text())
            assert status['status']=='complete'
            assert sha(path)==status['artifacts']['metrics.json']
            manifestpath=folder/'manifest.json'
            assert sha(manifestpath)==status['artifacts']['manifest.json']
            result=json.loads(path.read_text()); profile=result['profile']
            assert profile['seed']==seed and profile['variant']=='reference' and profile['epochs']==50
            protocols=result['evaluation']['protocols']
            for method in ('bf','pot'):
                pa,raw=protocols['pa_at_pa_selection'][method],protocols['no_pa_at_pa_selection'][method]
                assert pa['FP']==raw['FP'] and pa['TN']==raw['TN']
                assert pa['alpha']==raw['alpha'] and pa['threshold']==raw['threshold']
                gridrun=next(r for r in runs if r['dataset']==ds and int(r['seed'])==seed
                             and int(r['window'])==profile['window_sz'] and int(r['feedforward'])==profile['dim_feedforward'])
                np.testing.assert_allclose(pa['f1'],float(gridrun[method+'_f1']),atol=1e-7,rtol=0)
                for protocol in ('pa_at_pa_selection','no_pa_at_pa_selection','no_pa_optimized'):
                    values=protocols[protocol][method]
                    no_pa_runs.append({'dataset':ds,'configuration':config,'seed':seed,
                        'window':profile['window_sz'],'feedforward':profile['dim_feedforward'],
                        'method':method,'protocol':protocol,
                        **{k:values[k] for k in ('f1','precision','recall','TP','FP','TN','FN','alpha','threshold')},
                        'fpr':values['FP']/(values['FP']+values['TN'])})
            no_pa_sources.append({'file':str(path.relative_to(ROOT)),'sha256':sha(path),
                                  'manifest':str(manifestpath.relative_to(ROOT)),'manifest_sha256':sha(manifestpath)})
old.csv_write(DATA/'pa_no_pa_runs.csv',no_pa_runs)

def control_values(ds,config,method,protocol,metric):
    return np.array([100*r[metric] for r in no_pa_runs if r['dataset']==ds and r['configuration']==config
                     and r['method']==method and r['protocol']==protocol])

population_path = ROOT / 'results/analysis_opssat_window_cases/fragment_consensus.csv'
test_ids = {int(r['segment']) for r in readcsv(population_path)}
assert len(test_ids) == 529
rawpath = ROOT / 'data/raw/OPSSAT/segments.csv'
opstrace = []
ops_counts = {'magnetometer':0, 'photodiode':0}
ops_segments = {'magnetometer':set(), 'photodiode':set()}
magchannels = {'CADC0872','CADC0873','CADC0874'}
with rawpath.open() as f:
    for row in csv.DictReader(f):
        segment = int(row['segment'])
        if segment == 239:
            opstrace.append(row)
        if segment in test_ids and int(row['anomaly']):
            family = 'magnetometer' if row['channel'] in magchannels else 'photodiode'
            ops_counts[family] += 1
            ops_segments[family].add(segment)
assert len(opstrace) == 445 and sum(ops_counts.values()) == 28785
signal_sources.extend({'file':str(p.relative_to(ROOT)), 'sha256':sha(p)} for p in (rawpath,population_path))
old.csv_write(DATA / 'opssat_fragment_239.csv', opstrace)
old.csv_write(DATA / 'opssat_positive_population.csv', [
    {'family':k,'positive_fragments':len(ops_segments[k]),'positive_samples':n,
     'positive_sample_fraction':n/sum(ops_counts.values())} for k,n in ops_counts.items()])

def observations(fig, texts, y=.22):
    for i, text in enumerate(texts):
        fig.text(.07, y-i*.068, text, fontsize=19, color=INK)

def metrics_comparison(ds, configurations, texts):
    fig = canvas(ds+' · what changes in the metrics?')
    axes = [fig.add_axes([.075,.36,.39,.39]), fig.add_axes([.565,.36,.39,.39])]
    xs = np.arange(3)
    fields = ('precision','recall','f1')
    for ax, method, title in zip(axes, ('bf','pot'), ('Best F1 · 200 thresholds','POT')):
        for i, (w,h) in enumerate(configurations):
            c = lookup[ds,w,h]
            means = [c[method+'_'+m+'_mean'] for m in fields]
            sd = [c[method+'_'+m+'_sd'] for m in fields]
            positions = xs+(-.19 if i==0 else .19)
            ax.bar(positions, means, width=.32, color='#B8B8B8' if i==0 else RED,
                   label=f'W={w}, FF={h}', yerr=sd, capsize=3,
                   error_kw={'lw':1,'ecolor':'#555555'})
            for pos, mean, error in zip(positions,means,sd):
                ax.text(pos,mean+error+2,f'{mean:.1f}',ha='center',va='bottom',fontsize=14)
        ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100])
        ax.set_xticks(xs,['Precision','Recall','F1'])
        ax.set_ylabel('(%)');ax.set_title(title,loc='left',pad=15,fontsize=19)
        ax.grid(axis='y',color='#EEEEEE');ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.51,.875),fontsize=16)
    observations(fig,texts)
    footer(fig,'Mean ± sample SD · 3 seeds · 50 epochs · point adjustment')
    return fig

def ops_shapes():
    fig = canvas('OPS-SAT · brief departures inside longer fragments')
    ax = fig.add_axes([.075,.43,.52,.32])
    x = np.array([float(r['value']) for r in opstrace])
    ax.plot(np.arange(len(x)),x,color=INK,lw=1.4)
    ax.fill_between(np.arange(len(x)),0,1,where=x==0,transform=ax.get_xaxis_transform(),color=RED,alpha=.12)
    ax.set_xlabel('Sample index');ax.set_ylabel('Telemetry value')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax.set_title('Real magnetometer fragment · CADC0874',loc='left',fontsize=17,pad=12)
    ax.text(.02,.91,'Zero-valued stretches',transform=ax.transAxes,color=RED,fontsize=15)
    fraction = ops_counts['magnetometer']/sum(ops_counts.values())
    fig.text(.66,.72,f'{100*fraction:.0f}%',color=RED,fontsize=48)
    fig.text(.66,.64,'of positive-labelled samples\ncome from magnetometers',fontsize=18,linespacing=1.4)
    fig.text(.66,.50,'3 magnetometer channels\n6 photodiode channels',fontsize=17,color=GRAY,linespacing=1.5)
    observations(fig,[
        'Brief departures can sit inside an otherwise ordinary-looking fragment.',
        'A short window can change error peaks; extra history is not always helpful.',
    ],y=.235)
    footer(fig,'Fragment 239 · whole-fragment label, not exact anomaly onset · a possible mechanism, not a rule')
    return fig

def real_trace(ax,ds):
    example=real_examples[ds]
    t,x,y=example['indices'],example['values'],example['labels']
    ax.plot(t,x,color=INK,lw=1.5)
    edges=np.diff(np.r_[False,y,False].astype(int))
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        ax.axvspan(t[a]-.5,t[b-1]+.5,color=RED,alpha=.12)
    ax.set_title('Real interval · '+Path(example['track']).stem,loc='left',fontsize=18,pad=15)
    ax.set_xlabel('Sample index',fontsize=14)
    ax.set_ylabel('Telemetry (stored scale)',fontsize=15)
    ax.tick_params(labelsize=12)

def smap_context():
    fig = canvas('SMAP · extra history can contain repeated information')
    left=fig.add_axes([.07,.43,.42,.32]);right=fig.add_axes([.58,.43,.36,.32])
    real_trace(left,'SMAP')
    stats=[next(r for r in native_summary if r['dataset']=='SMAP' and r['label']==label) for label in (0,1)]
    values=[100*r['zero_step_fraction'] for r in stats]
    right.barh([1,0],values,color=['#B8B8B8',RED],height=.42)
    right.set_yticks([1,0],['Nominal','Anomalous']);right.set_xlim(0,100)
    right.set_xticks([0,50,100]);right.set_xlabel('Unchanged consecutive values (%)',fontsize=14)
    right.set_title('All test tracks · telemetry',loc='left',fontsize=17,pad=14)
    for y,v in zip([1,0],values):right.text(v+2,y,f'{v:.1f}',va='center',fontsize=16)
    observations(fig,[
        'About two thirds of successive values are identical, in both classes.',
        'Repeated values make redundancy plausible.',
    ],y=.235)
    footer(fig,'Real test data · shading = positive labels · telemetry coordinate only · example selection recorded in data/')
    return fig

def msl_changes():
    fig = canvas('MSL · anomalous intervals contain larger local changes')
    left=fig.add_axes([.07,.43,.42,.32]);right=fig.add_axes([.63,.43,.30,.32])
    real_trace(left,'MSL')
    stats=[next(r for r in native_summary if r['dataset']=='MSL' and r['label']==label) for label in (0,1)]
    values=[r['q95_abs_step_divided_by_train_range'] for r in stats]
    right.bar([0,1],values,color=['#B8B8B8',RED],width=.5)
    right.set_xticks([0,1],['Nominal','Anomalous'])
    right.set_ylim(0,.23);right.set_yticks([0,.1,.2])
    right.set_ylabel('95th percentile of |Δx|',fontsize=15)
    right.set_title('All test tracks',loc='left',fontsize=17,pad=14)
    for i,v in enumerate(values):right.text(i,v+.009,f'{v:.3f}',ha='center',fontsize=17)
    observations(fig,[
        'At the 95th percentile, local changes are about four times as large.',
    ],y=.235)
    footer(fig,'Real test data · shading = positive labels · right: |Δx| divided by nominal-training range')
    return fig

def pa_shapes():
    # Same 12-sample example as pa_example() in Results_II, reduced to one model.
    fig=canvas('Point adjustment · what changes after an event is hit?')
    ax=old.plain_axis(fig)
    truth=np.zeros(12,dtype=int);truth[2:8]=1
    direct=np.zeros(12,dtype=int);direct[[2,3,4,9]]=1
    adjusted=direct.copy();adjusted[2:8]=1
    x0,cw,gap,h=.285,.045,.009,.065
    for i in range(12):
        ax.text(x0+i*(cw+gap)+cw/2,.815,str(i+1),ha='center',fontsize=14,color=GRAY)
    for row,(name,values) in enumerate(zip(['True labels','Direct alarms','After PA'],[truth,direct,adjusted])):
        y=.70-row*.145
        ax.text(.065,y+h/2,name,va='center',fontsize=21)
        for i,value in enumerate(values):
            color='white'
            if value:color=RED if truth[i] else INK
            if row==2 and value and not direct[i]:color='#D3B8B4'
            ax.add_patch(Rectangle((x0+i*(cw+gap),y),cw,h,facecolor=color,edgecolor='#C9C4C2',lw=.8))
    for x,color,label in [(.285,RED,'Direct / true positive'),(.56,'#D3B8B4','Added by PA'),(.77,INK,'False alarm')]:
        ax.add_patch(Rectangle((x,.30),.016,.022,facecolor=color,edgecolor='none'))
        ax.text(x+.024,.311,label,va='center',fontsize=13,color=GRAY)
    ax.text(.285,.205,'Before PA:  TP = 3     FP = 1     FN = 3',fontsize=19)
    ax.text(.285,.135,'After PA:    TP = 6     FP = 1     FN = 0',fontsize=19)
    footer(fig,'Illustrative example · PA fills a detected positive interval; false alarms outside remain unchanged.')
    return fig

def takeaway():
    fig = canvas('What can we conclude about shorter windows?')
    for y,heading,text in [
        (.77,'Fewer false alarms can be a real benefit','But fewer alarms can also mean fewer directly detected anomalous samples.'),
        (.56,'PA changes what receives credit','A single crossing can credit a whole event; raw F1 counts individual samples.'),
        (.35,'Keep the two conclusions separate','Better PA F1 does not necessarily mean better point-by-point detection.'),
    ]:
        fig.text(.075,y,heading,fontsize=23,color=RED)
        fig.text(.075,y-.075,text,fontsize=18,color=INK)
    footer(fig,'Signal shape is a possible contributor; the matched controls show that method and evaluation also matter.')
    return fig

def raw_tradeoff():
    fig=canvas('Fewer false alarms can also mean fewer raw detections')
    axes=[fig.add_axes([.075,.35,.39,.40]),fig.add_axes([.565,.35,.39,.40])]
    for ax,metric,title in zip(axes,('fpr','recall'),('False-positive rate','Raw recall')):
        xs=np.arange(3)
        for i,config in enumerate(('reference','window_only')):
            arrays=[control_values(ds,config,'bf','no_pa_at_pa_selection',metric) for ds in DS]
            means=[v.mean() for v in arrays];sd=[v.std(ddof=1) for v in arrays]
            pos=xs+(-.19 if i==0 else .19)
            ax.bar(pos,means,.32,color='#B8B8B8' if i==0 else RED,label='Longer W' if i==0 else 'Shorter W',
                   yerr=sd,capsize=3,error_kw={'lw':1,'ecolor':'#555555'})
            for x,m,s in zip(pos,means,sd):ax.text(x,m+s+.22,f'{m:.2f}',ha='center',fontsize=13)
        ax.set_xticks(xs,['OPS-SAT\n10 → 5','SMAP\n10 → 5','MSL\n50 → 25'])
        ax.set_ylim(0,12);ax.set_ylabel('(%)');ax.set_title(title,loc='left',fontsize=20,pad=14)
        ax.grid(axis='y',color='#EEEEEE');ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.5,.87),fontsize=16)
    observations(fig,[
        'BF200: fewer false alarms, but also lower raw recall on OPS-SAT and SMAP.',
        'PA leaves false-positive counts unchanged; it credits more positive samples.',
    ],y=.19)
    footer(fig,'Same PA-selected α and threshold for each model · 3 matched seeds · reference architecture · FF unchanged')
    return fig

def no_pa_comparison(method):
    label='BF200' if method=='bf' else 'POT'
    fig=canvas(label+' · with and without point adjustment')
    axes=fig.subplots(1,3)
    fig.subplots_adjust(left=.07,right=.975,bottom=.29,top=.72,wspace=.25)
    protocols=('pa_at_pa_selection','no_pa_at_pa_selection','no_pa_optimized')
    for ax,ds in zip(axes,DS):
        xs=np.arange(3)
        for i,config in enumerate(('reference','window_only')):
            arrays=[control_values(ds,config,method,p,'f1') for p in protocols]
            means=[v.mean() for v in arrays];sd=[v.std(ddof=1) for v in arrays]
            positions=xs+(-.19 if i==0 else .19)
            ax.bar(positions,means,.32,color='#B8B8B8' if i==0 else RED,label='Longer W' if i==0 else 'Shorter W',
                   yerr=sd,capsize=2,error_kw={'lw':1,'ecolor':'#555555'})
            for x,m,s in zip(positions,means,sd):ax.text(x,m+s+2,f'{m:.1f}',ha='center',fontsize=12)
        profile=next(r for r in no_pa_runs if r['dataset']==ds and r['configuration']=='reference')
        short=next(r for r in no_pa_runs if r['dataset']==ds and r['configuration']=='window_only')
        ax.set_title(f"{ds}\nW {profile['window']} → {short['window']} · FF {profile['feedforward']}",loc='left',fontsize=17,pad=14)
        ax.set_xticks(xs,['PA','No PA\nsame settings','No PA\nnew selection']);ax.tick_params(axis='x',labelsize=10)
        ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('F1 (%)' if ds==DS[0] else '')
        ax.grid(axis='y',color='#EEEEEE');ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.55,.87),fontsize=16)
    text=('On OPS-SAT and SMAP, the PA gain does not become a raw F1 gain here.'
          if method=='bf' else 'Shorter windows improve raw F1 here, but it remains much lower than PA F1.')
    if method!='bf':
        observations(fig,[text],y=.19)
    fig.text(.055,.10,'Same settings: remove PA, keeping the α and threshold selected with PA.',fontsize=12,color=GRAY)
    footer(fig,('New selection: BF200 selects α and τ for raw F1 · same trained models · 3 seeds · 50 epochs'
                if method=='bf' else 'New selection: POT selects α for raw F1 using its per-α SPOT threshold · same models · 3 seeds'))
    return fig

def schema():
    fig = old.model_figure()
    fig.text(.055, .95, 'TranAD+', fontsize=27, color=RED, va='top')
    return fig

pages = [
    ('01_tranadplus', schema),
    ('02_benchmarks', benchmark),
    ('03_window', lambda: trends('window')),
    ('04_feedforward', lambda: trends('feedforward')),
    ('05_heatmap_bf200', lambda: heatmap('bf')),
    ('06_heatmap_pot', lambda: heatmap('pot')),
    ('07_opssat_metrics', lambda: metrics_comparison('OPS-SAT', [(10,8),(3,2)], [
        'Reducing W and FF together improves both F1 scores.',
        'BF200: higher precision, slightly lower recall. POT: recall also increases.'
    ])),
    ('08_opssat_shapes', ops_shapes),
    ('09_smap_metrics', lambda: metrics_comparison('SMAP', [(10,50),(3,50)], [
        'BF200 improves mainly through higher precision, with slightly lower recall.',
        'POT improves mainly through higher recall.'
    ])),
    ('10_smap_context', smap_context),
    ('11_msl_metrics', lambda: metrics_comparison('MSL', [(50,110),(13,110)], [
        'BF200 recall changes strongly; POT precision and recall stay almost unchanged.',
    ])),
    ('12_msl_changes', msl_changes),
    ('13_point_adjustment', pa_shapes),
    ('14_raw_tradeoff', raw_tradeoff),
    ('15_bf_pa_no_pa', lambda:no_pa_comparison('bf')),
    ('16_pot_pa_no_pa', lambda:no_pa_comparison('pot')),
    ('17_interpretation', takeaway),
]

extra_spec=importlib.util.spec_from_file_location('additional_results',OUT/'additional_results.py')
extra=importlib.util.module_from_spec(extra_spec)
extra_spec.loader.exec_module(extra)
print('Verifying architecture, weight-only and integer-precision reports...',flush=True)
pages.extend(extra.create_pages(ROOT,OUT,canvas,footer))

prs = Presentation()
prs.slide_width = Inches(old.W)
prs.slide_height = Inches(old.H)
prs.core_properties.author = 'Dario Liotta'
prs.core_properties.title = 'TranAD+ — benchmarks, ablation, architectures and quantization'
with PdfPages(OUT / 'Results.pdf', metadata={'Title': prs.core_properties.title, 'Author': 'Dario Liotta'}) as pdf:
    for i, (name, factory) in enumerate(pages, 1):
        print(f'[{i:02}/{len(pages)}] {name}', flush=True)
        fig = factory()
        # Verify that visible text stays inside the fixed slide canvas.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        visible_texts = list(fig.texts)
        for ax in fig.axes:
            visible_texts.extend([*ax.texts, ax.title, ax._left_title, ax._right_title,
                                  ax.xaxis.label, ax.yaxis.label])
        for text in visible_texts:
            if not text.get_visible() or not text.get_text():
                continue
            bounds = text.get_window_extent(renderer)
            # Locator tick objects outside the visible data interval are excluded.
            if bounds.x0 < -2 or bounds.x1 > fig.bbox.width+2 or bounds.y0 < -2 or bounds.y1 > fig.bbox.height+2:
                raise RuntimeError(f'Text outside slide {name}: {text.get_text()}')
        svg, png = FIG / (name+'.svg'), FIG / (name+'.png')
        fig.savefig(svg)
        fig.savefig(png, dpi=240)
        fig.savefig(FIG / (name+'.pdf'))
        pdf.savefig(fig)
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        old.add_svg(slide, svg, png, prs.part.package)
        plt.close(fig)
prs.save(OUT / 'Results.pptx')
# Remove only obsolete, regenerable figures from the previous version of this deck.
for stem in ('07_opssat','08_smap','09_msl','10_interpretation','14_interpretation'):
    for suffix in ('png','pdf','svg'):
        (FIG/(stem+'.'+suffix)).unlink(missing_ok=True)

old.csv_write(DATA / 'benchmarks.csv', [dict(dataset=ds, method=m,
    paper_f1_pct=100*evidence['paper_source']['values'][ds][m][0],
    ours_f1_pct=100*bench[ds,m]['metrics']['f1'], source=bench[ds,m]['source'],
    source_sha256=bench[ds,m]['sha256'], masked_diagnostic=(ds=='WADI' and m=='pot'))
    for ds in old.ORDER for m in ('pot','bf')])
old.csv_write(DATA / 'ablation_cells.csv', cells)
manifest = {
    'slides': len(pages), 'language': 'English', 'main_colour': RED,
    'benchmark_selection': 'Best available individual results per dataset and method; WADI POT is the explicitly marked masked diagnostic. Not seed averages.',
    'benchmark_sources': list(bench.values()),
    'benchmark_evidence': {'file': str(evidence_path.relative_to(ROOT)), 'sha256': sha(evidence_path)},
    'ablation': sources,
    'ablation_statistics': '270 runs, 90 cells, 3 seeds, 50 epochs; mean and sample SD; ordinary BF200/POT with PA; no dense grid.',
    'one_dimensional_slices': 'W varies at maximum FF; FF varies at maximum W; labelled dataset-specific vertical scales to show within-dataset trends.',
    'heatmap_limits_f1_pct': heatmap_limits,
    'heatmap_colours': 'Separate labelled normalization and colorbar for each dataset; compare within-dataset trends using colours and cross-dataset levels using numeric labels.',
    'signal_sources': signal_sources,
    'signal_summary': native_summary,
    'real_examples': {ds:{k:v for k,v in example.items() if k not in ('indices','values','labels')} for ds,example in real_examples.items()},
    'no_pa_sources': no_pa_sources,
    'no_pa_design': 'FP32 reference architecture only; W10/W5 for OPS-SAT and SMAP, W50/W25 for MSL, same FF, seeds 2026–2028, 50 epochs. PA results numerically verified against corresponding original grid runs. Same-selection controls keep alpha/threshold fixed; raw-optimized controls select alpha/threshold for raw F1 (POT retains per-alpha SPOT thresholds). No retraining.',
    'opssat_positive_samples': ops_counts,
    'examples': 'All three dataset traces are real measurements from saved telemetry coordinate 0, with actual supplied labels. Selection rules/candidates are saved; examples illustrate signal shapes and are not random representative samples or physical fault diagnoses. Only the PA educational diagram is synthetic.',
    'signal_methods': 'Telemetry coordinate only in native datasets; absolute consecutive differences within tracks, divided by nominal-training range, excluding track-boundary differences. OPS-SAT population is the fixed 529-fragment test set. No new training or model inference.',
    'schema': {'source_builder': str(Path(old.__file__).relative_to(ROOT)), 'sha256': sha(old.__file__), 'function': 'model_figure; reused unchanged'},
    'interpretation': 'Observed effects are distinguished from possible mechanisms. No claim of demonstrated overfitting or a specific physical fault.',
    'builder_sha256': sha(__file__),
    'additional_results': {'file':'data/additional_results_provenance.json',
                           'sha256':sha(DATA/'additional_results_provenance.json')},
    'outputs': [{'file':str(p.relative_to(OUT)), 'sha256':sha(p)} for p in sorted(FIG.iterdir())],
}
(DATA / 'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
(OUT / 'README.md').write_text('''# Benchmarks and ablation discussion

Open `Results.pdf` or `Results.pptx` (35 slides). No cover page. English, Padova red, vector PDF/SVG figures with high-resolution PowerPoint fallbacks. Heatmaps use the exact first-presentation red palette and flat vector cells.

1. Original presentation network schematic, reused unchanged.
2. Previously requested best individual benchmark values. WADI POT 53.39% is marked as the NaN-mask diagnostic; no data rows or labels are dropped. MSL POT 90.17% is the individual maximum, not the closest-to-paper run used in the thesis evidence. SMD POT is 68.61% from V16.
3–4. Window and FF sweeps, mean ± sample SD over 3 seeds, 50 epochs. The other parameter is fixed at its largest tested value. Labelled dataset-specific vertical scales show within-dataset trends.
5–6. All 30 configurations per dataset; three separately labelled colour scales, so small within-dataset trends remain visible.
7, 9, 11. Grouped bar charts replace the previous F1-only tables. Precision, recall and F1 are all shown with mean ± sample SD for both BF200 and POT. OPS-SAT compares W10/FF8 and W3/FF2; SMAP W10 and W3 at FF50; MSL W50 and W13 at FF110.
8. Real OPS-SAT magnetometer fragment 239 with zero-valued stretches; measured positive-sample population share, not channel counts alone.
10. A real SMAP trace, with actual positive-label shading, alongside the within-track equal-value fractions for nominal and positive labels. The event is selected by median length among positive runs of at least 20 samples with at least 50% identical successive values.
12. A real MSL trace, with actual positive-label shading, alongside pooled MSL 95th-percentile absolute steps (scaled by nominal-training range). The event is selected by median length among positive runs of at least 20 samples with within-event step q95 above pooled nominal q95. These are descriptive examples, not representative samples, causal explanations of F1, or expert fault classes.
13. A simplified version of the 12-sample PA example from the second presentation: true labels, direct alarms and adjusted alarms. Added positives and an unchanged false alarm are explicitly distinguished; TP/FP/FN counts are shown before and after PA.
14. Raw false-positive rate and recall for the matched window comparison at each model's saved PA-selected BF200 alpha and threshold. FP/TN counts are verified identical before/after PA.
15–16. BF200 and POT with PA, without PA at fixed selection, and without PA after raw-F1 selection. These use exactly the same trained FP32 reference models within each comparison: W10/W5 for OPS-SAT and SMAP, W50/W25 for MSL, FF fixed, seeds 2026–2028, 50 epochs. The PA results numerically match the corresponding original grid runs. POT chooses alpha among its per-alpha SPOT thresholds; it does not freely sweep thresholds like BF200. Status/metric/manifest hashes are verified. No architectural variant or quantized model is included.
17. Short interpretation distinguishing false alarms, event credit and direct pointwise detection.
18–19. Three architecture diagrams and the four W/FF settings; 240 trained models, five seeds per cell, 50 epochs.
20–21. All four architectures at reference W/FF: F1 with PA and separately optimized F1 without PA, mean ± sample SD over five seeds.
22–24. Per-dataset precision, recall, F1 and binary AUC-PA for all four architectures and BF200/POT, at reference W/FF.
25. Unique parameters and analytical matrix MACs relative to TranAD+ at reference W/FF.
26–27. BF200/POT architecture-by-W/FF grids; five seeds per cell, all 240 trained models included.
28. Weight-only quantization results: quantized storage with FP32 calculations.
29. Integer matrix-product quantization results, W8A8 and W4A4. W4A4 uses INT8 instructions, not native INT4.
30–31. Integer precision by architecture, BF200/POT with PA; each cell averages four W/FF settings and five seeds.
32. Integer precision without PA, optimized for raw F1.
33. Stored tensor payloads in KiB at reference W/FF, including scales/biases/buffers, for every architecture and dataset.
34–35. Integer precision/recall/F1/AUC-PA, averaged over the declared 80 parents per dataset (four architectures × four settings × five seeds).

The appended architecture and precision section contains results only, with no new causal explanations. Selected fixed-setting comparisons and pooled-over-setting comparisons are explicitly separated. V16/V17 metrics and V18 raw-optimized metrics are hash-checked against complete status artifacts; exact run tables and source manifests are under data/. No new training or inference is performed.

The first ablation plots use point adjustment. The added controls explicitly separate PA and no-PA metrics and their optimization criteria. All results use ordinary BF200 and POT, not the dense grid. Ablation results are means, whereas the benchmark table shows selected individual runs. The disjoint V10/V12 cells remain a descriptive combined grid; source-revision differences are recorded in provenance rather than certified numerically equivalent here. The newly added no-PA window comparisons have explicitly labelled window sizes, which need not equal the extreme configurations contrasted on slides 7/9/11.

`data/benchmarks.csv`, `data/ablation_cells.csv`, and `data/provenance.json` retain exact displayed values, source paths and hashes. The build checks the 270 original report hashes and recomputes plotted means/SDs. `native_signal_summary.csv`, `opssat_positive_population.csv`, `opssat_fragment_239.csv`, `smap_real_example.csv`, `msl_real_example.csv`, `example_candidates.csv` and `pa_no_pa_runs.csv` retain the measurements, selection population and paired controls. Native descriptive statistics are pooled/sample-weighted, may reflect channel composition and are not claimed to identify physical fault causes. All signal source hashes are recorded. Original presentations, thesis files, source results and training code are preserved.

Rebuild: `python -B presentations/2026-11-11_results_discussion/build.py` from the repository root. Dependencies: NumPy, matplotlib, python-pptx, lxml (the existing local presentation dependency directory is supported).
''')
print('Completed:', OUT / 'Results.pdf')
