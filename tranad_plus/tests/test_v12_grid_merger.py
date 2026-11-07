"""Grid parser and 6x5 coverage contract; no training, no result files needed."""
from campaign_v12 import BASES,GRID,HISTORICAL,SEEDS,new_pairs
from tools.merge_v10_v12 import attach_levels
import pandas as pd


def test_v10_to_effective_dimensions():
    levels=(25,50,75,100)
    entries=[]
    for ds in BASES:
        prefix=ds.lower().replace('-','')
        for ix, w in enumerate(levels):
            for jx, h in enumerate(levels):
                for seed in SEEDS:
                    entries.append({'dataset':ds,'identifier':f'toast10_{prefix}_w{w:03d}_h{h:03d}_s{seed}',
                                    'expected_window':HISTORICAL[ds]['windows'][ix],
                                    'expected_ff':HISTORICAL[ds]['feedforwards'][jx]})
    parsed=attach_levels(pd.DataFrame(entries),'v10')
    assert len(parsed)==144
    assert (parsed['window']==parsed.expected_window).all()
    assert (parsed['feedforward']==parsed.expected_ff).all()


def test_combination_exact_without_duplicates():
    rows=[]
    for ds in BASES:
        prefix=ds.lower().replace('-','')
        for w,h in new_pairs(ds):
            for seed in SEEDS:
                rows.append({'dataset':ds,'identifier':f'toast12_{prefix}_w{w:03d}_h{h:03d}_s{seed}'})
    result=attach_levels(pd.DataFrame(rows),'v12')
    assert len(result)==126
    for ds in BASES:
        subset=result[result.dataset==ds]
        assert set(zip(subset.window,subset.feedforward))==set(new_pairs(ds))
        assert len(subset)==42
