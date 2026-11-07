"""V12 campaign integrity, dataset-independent console and output isolation."""
from pathlib import Path
import json
from campaign_v12 import make_plan, GRID, BASES, SEEDS, HISTORICAL, ADDITIONS, new_pairs
from console_telemetry import epoch_line, evaluation_line
from launch_v12 import validated_plan


def test_frozen_plan():
    plan=validated_plan()
    assert plan==make_plan()
    rows=plan['experiments']
    assert len(rows)==126 and len(set(e['identifier'] for e in rows))==126
    for dataset,base in BASES.items():
        items=[r for r in rows if r['base_profile']==base['base']]
        assert len(items)==14*len(SEEDS)
        assert {r['batch_sz'] for r in items}=={base['batch']}
        assert {r['epochs'] for r in items}=={50}
        assert {r['seed'] for r in items}==set(SEEDS)
    assert len({r['identifier'] for r in rows})==126


def test_uniform_stdout():
    for dataset in ('OPS-SAT','SMAP','MSL'):
        line=epoch_line(dataset=dataset, identifier='id',epoch=7,epochs=50,
                        loss=.2,lr=.001,seconds=1,batches=10,windows=100)
        assert '[TRAIN]' in line and '007/050' in line and dataset in line
        eval_line=evaluation_line(dataset=dataset, identifier='id',index=7,total=21,alpha=.3)
        assert '[EVAL]' in eval_line and '07/21' in eval_line and dataset in eval_line


def test_full_factorial_grid_extension():
    expected={'OPS-SAT':((4,6),(3,)), 'SMAP':((4,6),(32,)), 'MSL':((32,44),(69,))}
    for ds,(ws,hs) in expected.items():
        assert ADDITIONS[ds]['windows']==ws
        assert ADDITIONS[ds]['feedforwards']==hs
        old={(w,f) for w in HISTORICAL[ds]['windows'] for f in HISTORICAL[ds]['feedforwards']}
        new=set(new_pairs(ds))
        complete={(w,f) for w in GRID[ds]['windows'] for f in GRID[ds]['feedforwards']}
        assert len(old)==16 and len(new)==14 and len(complete)==30
        assert not old & new
        assert old | new == complete


def test_training_protocol_matches_v10():
    rows=make_plan()['experiments']
    assert {r['validation_fraction'] for r in rows if r['base_profile']==BASES['OPS-SAT']['base']}=={0.2}
    assert {r['split_seed'] for r in rows if r['base_profile']==BASES['OPS-SAT']['base']}=={2026}
    for ds in ('OPS-SAT','SMAP','MSL'):
        selected=[r for r in rows if r['base_profile']==BASES[ds]['base']]
        assert {r['batch_sz'] for r in selected}=={BASES[ds]['batch']}
        assert {r['seed'] for r in selected}==set(SEEDS)

