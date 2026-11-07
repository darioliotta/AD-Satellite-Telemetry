"""Drop-in upstream evaluation with cached TranAD decoder outputs.

21 alphas and 200 thresholds are still evaluated exactly as in the paper. The
GPU performs one train/test decoder pass instead of 21 repeated passes. A
separate, optional dense Best-F1 oracle is annotated in the BF pickle.
"""
from __future__ import annotations
import functools
import pickle
import runpy
import sys
from pathlib import Path

from TranADPlus.src import helpers


def main():
    from gpu_settings import configure
    configure()
    script=Path(__file__).resolve().parent/'TranADPlus'/'scripts'/'evaluation.py'
    enabled='--alpha_search' in sys.argv and sys.argv[sys.argv.index('--alpha_search')+1].lower() in ('true','1','yes')
    total=21 if enabled else 1
    cache={}
    best_dense=None
    counter=0
    original=helpers.evaluate_TranAD

    @functools.wraps(original)
    def run(model,train_dataloader,test_dataloader,alpha=0.5,beta=0.5,pot_q=1e-5,device='cpu'):
        nonlocal best_dense,counter
        counter+=1
        from console_telemetry import emit_evaluation
        from config import get_profile
        profile=get_profile(sys.argv[sys.argv.index('--identifier')+1])
        emit_evaluation(dataset=profile['dataset'], identifier=profile['identifier'],
                        index=counter, total=total, alpha=float(alpha))
        if abs(alpha+beta-1)>1e-10:
            raise ValueError('v11 evaluation requires beta=1-alpha')
        from paper_metrics import decoder_arrays, native_metrics_from_arrays
        if 'data' not in cache:
            print('Collecting decoder outputs once for the complete alpha grid...',flush=True)
            cache['data']=(decoder_arrays(model,train_dataloader,device,
                f'EVAL {profile["dataset"]} TRAIN-SCORES'),
                decoder_arrays(model,test_dataloader,device,
                f'EVAL {profile["dataset"]} TEST-SCORES'))
        result,dense=native_metrics_from_arrays(*cache['data'],alpha,pot_q)
        if best_dense is None or dense['bf_dense_f1']>best_dense['bf_dense_f1']:
            best_dense={**dense,'bf_dense_alpha':float(alpha),'bf_dense_beta':float(beta)}
        return result

    helpers.evaluate_TranAD=run
    try:
        sys.argv[0]=str(script)
        runpy.run_path(str(script),run_name='__main__')
        # Original evaluation.py saves BF and POT pickles, with their 21-alpha
        # selection retained. Attach dense oracle to the BF result only.
        if best_dense is not None:
            from TranADPlus.src.global_config import config
            idx=sys.argv.index('--identifier')
            identifier=sys.argv[idx+1]
            destination=Path(config.pickle_dir)/f'{identifier}_bf.pkl'
            with destination.open('rb') as stream:bf=pickle.load(stream)
            bf.update(best_dense)
            temporary=destination.with_suffix('.v11.tmp')
            with temporary.open('wb') as stream:pickle.dump(bf,stream)
            temporary.replace(destination)
    finally:
        helpers.evaluate_TranAD=original


if __name__=='__main__':main()
