"""V5 ablation definitions and *inference-only* decoder removal experiment.

This module does not alter upstream TranAD+ or silently change its loss.
First-stage inference uses identical trained encoder, first decoder and head;
it omits the second decoder and the self-conditioning second encoder pass.
It is an INFERENCE ablation on the trained Normal, not a retrained Light.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class FirstStageOnly(nn.Module):
    """Exact original first reconstruction; report two identical views for scorers."""
    def __init__(self, original):
        super().__init__()
        self.n_feats = original.n_feats
        self.window_sz = original.window_sz
        self.batch_sz = original.batch_sz
        self.pos_encoder = original.pos_encoder
        self.transformer_encoder = original.transformer_encoder
        self.transformer_decoder1 = original.transformer_decoder1
        self.fcn = original.fcn

    def forward(self, src, tgt):
        conditioning = torch.zeros_like(src)
        x = torch.cat((src, conditioning), dim=2) * math.sqrt(self.n_feats)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)
        tgt = tgt.repeat(1, 1, 2)
        x1 = self.fcn(self.transformer_decoder1(tgt, memory))
        return x1, x1


def assert_stage1_parity(original, first_stage, sample, device, atol=3e-6):
    original.eval(); first_stage.eval()
    source=sample.to(device)
    target=source[-1:, :, :]
    with torch.inference_mode():
        first,_=original(source,target)
        only,_=first_stage(source,target)
    delta=float((first-only).abs().max().item())
    if not torch.allclose(first,only,atol=atol,rtol=1e-5):
        raise AssertionError(f'First-stage implementation mismatch, max_abs_error={delta}')
    return delta


def parse_ablation(profile):
    """Describe controlled factor changes, including reference vs architectural note."""
    return {
        'family': profile.get('ablation_family', 'unspecified'),
        'factor': profile.get('ablation_factor', 'baseline'),
        'level': profile.get('ablation_level'),
        'reference': profile.get('ablation_reference'),
        'seed': profile.get('seed'),
        'trained_architecture': 'original_TranADPlus_2stage',
        'hyperparameters': {'window_sz': profile['window_sz'],
                            'dim_feedforward': profile['dim_feedforward'],
                            'num_layers': profile['num_layers']},
        'caution': 'Window and feedforward sweep retrains original TranAD+; '
                   'first-stage-only is evaluated separately on the SAME frozen trained weights.'
    }


def stage1_benchmark(original, sample, device, settings):
    from observability.measure_experiment import benchmark, architecture_statistics
    wrapper=FirstStageOnly(original).to(device).eval()
    delta=assert_stage1_parity(original,wrapper,sample,device)
    profile=benchmark(wrapper, torch.device(device), sample, int(settings['warmup']),int(settings['iterations']))
    profile['includes_both_decoders']=False
    profile['mode']='torch_inference_mode_one_stage_forward_only'
    return wrapper, {'status':'measured',
                     'scope':'post_training_inference_only_no_retraining',
                     'first_stage_parity_max_abs_error':delta,
                     'model_statistics':architecture_statistics(wrapper),
                     'inference':profile}


def evaluate_native_stage1(original, meta, dataset_name, device, settings, report_dir, identifier):
    """Frozen checkpoint -> first decoder only scores on native data."""
    import numpy as np
    from pathlib import Path
    from TranADPlus.src.datasets import TSDataset_tracks
    from observability.export_anomaly_scores import pair_paths
    from pipeline_metrics import _scores, _raw_quality
    base = __import__('project_paths').PREPROCESSED_ROOT / dataset_name
    tr_paths,tr_labels,_ = pair_paths(base,'Train')
    te_paths,te_labels,_ = pair_paths(base,'Test')
    ws = int(meta['window_sz']); hp=meta['hyperparameters']
    train_data=TSDataset_tracks(track_paths=tr_paths, anom_paths=tr_labels,window_size=ws,
                                 scaling=True,padding=bool(hp.get('padding',True)))
    test_data=TSDataset_tracks(track_paths=te_paths, anom_paths=te_labels,window_size=ws,
                                scaling=True,scaler=train_data.scaler,padding=bool(hp.get('padding',True)))
    sample = torch.tensor(np.random.default_rng(971).random((ws,1,int(meta['features']))),dtype=torch.float32)
    wrapper, measured = stage1_benchmark(original,sample,device,settings)
    train_scores, _,train_perf=_scores(wrapper,train_data,torch.device(device),int(settings['score_batch_size']),1.0)
    test_scores,labels,test_perf=_scores(wrapper,test_data,torch.device(device),int(settings['score_batch_size']),1.0)
    threshold=float(np.quantile(train_scores,float(settings['threshold_train_quantile'])))
    measured['quality']={**_raw_quality(test_scores,labels,threshold),
             'threshold_source':'training_scores_quantile',
             'quantile':float(settings['threshold_train_quantile']),
             'score_formula':'mean((decoder1-target)^2, axis=features)',
             'scoring_performance':{'train':train_perf,'test':test_perf}}
    return measured


def evaluate_opssat_stage1(original,train_data,test_data,test_fragments,device,settings):
    """Use identical weak labels, fragment boundaries and fixed threshold as v4."""
    import numpy as np
    from opssat_pipeline import score_decoders, adjust_per_fragment, point_metrics, _auc
    sample=test_data.windows[:1].permute(1,0,2)
    wrapper,measured=stage1_benchmark(original,sample,device,settings)
    first_tr,_,targets_tr,perf_tr=score_decoders(wrapper,train_data,device,int(settings['score_batch_size']))
    first_te,_,targets_te,perf_te=score_decoders(wrapper,test_data,device,int(settings['score_batch_size']))
    tscores=((first_tr-targets_tr)**2).reshape(-1)
    scores=((first_te-targets_te)**2).reshape(-1)
    threshold=float(np.quantile(tscores,float(settings['threshold_train_quantile'])))
    labels=test_data.point_labels
    from pipeline_metrics import _raw_quality
    raw=_raw_quality(scores,labels,threshold)
    adjusted=adjust_per_fragment(scores,labels,test_fragments,threshold)
    offset=0;seg_scores=[];seg_predict=[];seg_labels=[]
    for frag in test_fragments:
        n=len(frag['values']);frag_scores=scores[offset:offset+n];offset+=n
        seg_scores.append(float(frag_scores.max()))
        seg_predict.append(int(np.any(frag_scores>threshold)))
        seg_labels.append(int(frag['label']))
    seg_labels=np.asarray(seg_labels,dtype=np.int8)
    measured['quality']={**raw,'score_formula':'(decoder1-target)^2',
                        'threshold_source':'training_scores_quantile',
                        'point_adjusted':point_metrics(labels,adjusted,'point_adjusted'),
                        'fragment_metrics':{**point_metrics(seg_labels,seg_predict,'segment'),
                                            **_auc(seg_labels,np.asarray(seg_scores))},
                        'scoring_performance':{'train':perf_tr,'test':perf_te},
                        'label_protocol':'weak fragment labels; adjustment only within fragment'}
    return measured
