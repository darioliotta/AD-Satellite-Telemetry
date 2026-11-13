"""Compact seed summaries and paired precision comparisons against FP32."""
import argparse
import json
from pathlib import Path
import statistics

from tranad_plus.core.common import hash_file
from tranad_plus.campaigns.temporal.summarize import write_csv


def summarize(root):
    root=Path(root);rows=[];unavailable=[]
    for p in sorted(root.rglob('status.json')):
        status=json.loads(p.read_text())
        if status['status']!='complete':continue
        metrics=p.parent/'metrics.json'
        if hash_file(metrics)!=status['artifacts']['metrics.json']:raise RuntimeError('Changed metrics: '+str(metrics))
        report=json.loads(metrics.read_text());profile=report['profile']
        checkpoints=list(report.get('milestones',{}).values()) if report['stage']=='qat' else [{'extra_epochs':0,'formats':report['formats']}]
        for checkpoint in checkpoints:
          extra=checkpoint['extra_epochs']
          for name,result in checkpoint['formats'].items():
            if result.get('status')=='not_available':
                unavailable.append({'stage':report['stage'],'kind':report['kind'],'dataset':profile['dataset'],
                                    'variant':profile['variant'],'configuration':profile['configuration'],
                                    'extra_epochs':extra,'seed':profile['seed'],'format':name,'reason':result['reason']})
                continue
            for protocol,key in [('oracle','oracle'),('parent_locked','parent_locked')]:
                for method,item in result[key].items():
                    if method not in ('bf','pot'):continue
                    row={'stage':report['stage'],'kind':report['kind'],'dataset':profile['dataset'],
                         'variant':profile['variant'],'configuration':profile['configuration'],'extra_epochs':extra,'seed':profile['seed'],
                         'window':profile['window_sz'],'ff':profile['dim_feedforward'],'format':name,
                         'protocol':protocol,'method':method,'alpha':item.get('alpha'),'threshold':item.get(method+'_threshold')}
                    for label,suffix in [('f1_pct','f1'),('precision_pct','precision'),('recall_pct','recall'),('AUC_PA_pct','ROC/AUC')]:
                        value=item.get(method+'_'+suffix);row[label]=None if value is None else 100*value
                    reference=report.get('author_reference',{}).get('reference_pot_f1') if method=='pot' else None
                    row['reference_f1_pct']=100*reference if reference is not None else None
                    row['delta_from_reference_pp']=(row['f1_pct']-row['reference_f1_pct']
                                                    if row['f1_pct'] is not None and reference is not None else None)
                    value=item.get('raw_without_PA',{}).get(method+'_f1');row['raw_f1_pct']=None if value is None else 100*value
                    r=result['resources'];s=result['storage'];op=r.get('analytic_operations',{})
                    row.update(tensor_bytes=s['tensor_bytes_including_bias_scales_buffers'],file_bytes=s['artifact_file_bytes'],
                               matrix_coverage=s['matrix_coverage_fraction'],resident_parameter_bytes=r['parameter_bytes_resident'],
                               execution=s['execution'],median_forward_ms=r['median_ms'],
                               mean_forward_ms=r.get('mean_ms'),p95_forward_ms=r['p95_ms'],
                               analytic_MACs=op.get('matmul_MACs_per_window'),analytic_FLOPs=op.get('matmul_FLOPs_per_window'))
                    rows.append(row)
    write_csv(root/'tables/per_seed.csv',rows);write_csv(root/'tables/unavailable_formats.csv',unavailable)
    keys=('stage','kind','dataset','variant','configuration','extra_epochs','window','ff','format','protocol','method','execution')
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    summary=[]
    for key,values in sorted(groups.items()):
        if len({r['seed'] for r in values})!=len(values):raise RuntimeError('Duplicate seed in result group')
        r=dict(zip(keys,key));r['model_seeds']=len(values)
        for metric in ('f1_pct','precision_pct','recall_pct','AUC_PA_pct','raw_f1_pct','tensor_bytes','file_bytes','median_forward_ms','analytic_MACs','analytic_FLOPs'):
            a=[v[metric] for v in values if v[metric] is not None]
            r[metric+'_available_seeds']=len(a);r[metric+'_mean']=statistics.mean(a) if a else None
            r[metric+'_std']=statistics.stdev(a) if len(a)>1 else None
        summary.append(r)
    write_csv(root/'tables/mean_std.csv',summary)
    identity=('dataset','variant','configuration','extra_epochs','seed','format','protocol','method')
    index={(r['stage'],r['kind'])+tuple(r[k] for k in identity):r for r in rows}
    deltas=[]
    for r in rows:
        if r['stage']=='ptq' and r['format']!='fp32':
            values={**r,'format':'fp32'};other=index.get(('ptq','ptq')+tuple(values[k] for k in identity))
            if other is not None:
                delta={k:r[k] for k in identity}
                delta.update(comparison='PTQ_vs_FP32',quantized_f1_pct=r['f1_pct'],baseline_f1_pct=other['f1_pct'],
                             tensor_compression_ratio=other['tensor_bytes']/r['tensor_bytes'] if r['tensor_bytes'] else None,
                             file_compression_ratio=other['file_bytes']/r['file_bytes'] if r['file_bytes'] else None)
                for metric in ('f1_pct','precision_pct','recall_pct','AUC_PA_pct','raw_f1_pct'):
                    delta[metric+'_delta_pp']=r[metric]-other[metric] if r[metric] is not None and other[metric] is not None else None
                delta['delta_pp']=delta['f1_pct_delta_pp'];deltas.append(delta)
        if r['kind']!='qat':continue
        for label,prefix,extra in [('initial_PTQ',('ptq','ptq'),0),('matched_FP32_training_then_PTQ',('qat','control'),r['extra_epochs'])]:
            values={**r,'extra_epochs':extra};other=index.get(prefix+tuple(values[k] for k in identity))
            if other is None:continue
            value=None if r['f1_pct'] is None or other['f1_pct'] is None else r['f1_pct']-other['f1_pct']
            deltas.append({k:r[k] for k in identity}|{'comparison':label,'QAT_f1_pct':r['f1_pct'],
                          'control_f1_pct':other['f1_pct'],'delta_pp':value})
    write_csv(root/'tables/paired_deltas.csv',deltas)
    paired_groups={}
    paired_keys=('dataset','variant','configuration','extra_epochs','format','protocol','method','comparison')
    for r in deltas:paired_groups.setdefault(tuple(r[k] for k in paired_keys),[]).append(r)
    paired_summary=[]
    for key,values in sorted(paired_groups.items()):
        r=dict(zip(paired_keys,key));r['model_seeds']=len(values)
        for metric in ('delta_pp','precision_pct_delta_pp','recall_pct_delta_pp','AUC_PA_pct_delta_pp','tensor_compression_ratio','file_compression_ratio'):
            a=[v[metric] for v in values if v.get(metric) is not None]
            r[metric+'_mean']=statistics.mean(a) if a else None
            r[metric+'_std']=statistics.stdev(a) if len(a)>1 else None
        paired_summary.append(r)
    write_csv(root/'tables/paired_mean_std.csv',paired_summary)
    return {'metric_rows':len(rows),'summary_groups':len(summary),'paired_comparisons':len(deltas),'unavailable_formats':len(unavailable)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[3]/'results/v16')
    print(summarize(p.parse_args().output_root))
