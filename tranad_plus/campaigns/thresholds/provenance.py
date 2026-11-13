"""Read-only provenance checks against completed V17 inputs and models."""
import json
from pathlib import Path
import numpy as np
from tranad_plus.core.common import hash_file,digest
from tranad_plus.core.bootstrap import BASELINE_ROOT

ROOT_NUMERIC={'gpu_batches.py','gpu_settings.py','opssat_pipeline.py','opssat_scoring.py',
              'paper_metrics.py','project_paths.py','config.py','pipeline_metrics.py',
              'campaigns/integer/runtime.py','campaigns/integer/kernel.py','campaigns/integer/_cuda.py'}


def numerical_sources(previous,root=BASELINE_ROOT):
    selected={name:expected for name,expected in previous.items()
              if name in ROOT_NUMERIC or name.startswith('TranADPlus/')
              or (name.startswith('core/') and '/tests/' not in name)}
    required={'core/data.py','core/evaluation/scoring.py','TranADPlus/src/models.py',
              'campaigns/integer/runtime.py','campaigns/integer/kernel.py','campaigns/integer/_cuda.py'}
    if not required<=selected.keys():raise RuntimeError('V17 manifest lacks numerical source hashes')
    for name,expected in selected.items():
        if hash_file(Path(root)/name)!=expected:raise RuntimeError('Numerical implementation changed since V17: '+name)
    return selected


def verify_source(folder):
    folder=Path(folder)
    status=json.loads((folder/'status.json').read_text())
    if status['status']!='complete':raise RuntimeError('V17 source is incomplete: '+str(folder))
    for name,expected in status['artifacts'].items():
        path=folder/name
        if not path.resolve().is_relative_to(folder.resolve()):raise RuntimeError('Invalid artifact path')
        if hash_file(path)!=expected:raise RuntimeError('Changed V17 artifact: '+str(path))
    report=json.loads((folder/'metrics.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
    if report['signature']!=status['signature'] or digest(manifest['binding'])!=status['signature']:
        raise RuntimeError('V17 source signatures differ')
    if report['profile']!=manifest['binding']['profile']:raise RuntimeError('V17 source profiles differ')
    return report,manifest


def assert_score_statistics(grid,records,reference,rtol,atol):
    old=reference['score_statistics_by_split']['test']
    if len(old)!=len(records) or len(grid)!=len(old):raise RuntimeError('Alpha grid changed')
    current=[]
    for scores,record,saved in zip(grid,records,old):
        if record['alpha']!=saved['alpha']:raise RuntimeError('Alpha row order changed')
        measured={'alpha':record['alpha'],'mean':float(scores.mean()),'std':float(scores.std()),
                  'min':float(scores.min()),'max':float(scores.max()),
                  'quantiles_50_90_99_999':np.quantile(scores,[.5,.9,.99,.999]).tolist()}
        for key in ('mean','std','min','max','quantiles_50_90_99_999'):
            np.testing.assert_allclose(measured[key],saved[key],rtol=rtol,atol=atol,
                                       err_msg='V17 test score statistics differ: '+key)
        current.append(measured)
    return {'score_statistics_within_v17_tolerance':True,'rtol':rtol,'atol':atol,
            'scope':'Aggregate statistics only; full V17 pointwise score arrays were not archived',
            'statistics':current}
