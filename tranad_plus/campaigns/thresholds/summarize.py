"""Comparable per-seed tables, paired selection deltas and five-seed summaries."""
import json
from pathlib import Path
import statistics
from tranad_plus.core.common import hash_file
from tranad_plus.campaigns.temporal.summarize import write_csv


def summarize(root):
    root=Path(root);rows=[];pairs=[]
    for path in sorted(root.glob('*/*/*/seed_*/*/status.json')):
        state=json.loads(path.read_text())
        if state['status']!='complete':continue
        if hash_file(path.parent/'metrics.json')!=state['artifacts']['metrics.json']:
            raise RuntimeError('Changed result metrics: '+str(path))
        report=json.loads((path.parent/'metrics.json').read_text());profile=report['profile']
        for protocol,methods in report['evaluation']['protocols'].items():
            for method,endpoint in methods.items():
                row={k:profile[k] for k in ('dataset','variant','configuration','seed','window_sz','dim_feedforward')}
                row.update(format=report['format'],protocol=protocol,method=method,
                           alpha=endpoint['alpha'],threshold=endpoint['threshold'],
                           point_adjustment=endpoint['point_adjustment'],
                           selection_objective='no_PA' if protocol in ('no_pa_optimized','pa_at_no_pa_selection') else 'PA',
                           auc_name='AUC-PA' if endpoint['point_adjustment'] else 'AUC-bin')
                for key in ('f1','precision','recall','auc_binary'):
                    value=endpoint[key];row[key+'_pct']=100*value if value is not None else None
                for key in ('TP','TN','FP','FN'):row[key]=endpoint[key]
                row.update(all_positive_f1_pct=100*report['evaluation']['all_positive_baseline']['f1'],
                           test_inference_seconds=report['inference']['seconds_including_transfer'],
                           score_batch_size=report['inference']['batch_size'])
                rows.append(row)
        ev=report['evaluation']['protocols']
        for method in ('bf','pot'):
            old,new=ev['no_pa_at_pa_selection'][method],ev['no_pa_optimized'][method]
            pairs.append({**{k:profile[k] for k in ('dataset','variant','configuration','seed')},'format':report['format'],'method':method,
                          'raw_f1_at_PA_selection_pct':100*old['f1'],'raw_f1_optimized_pct':100*new['f1'],
                          'raw_f1_gain_from_reselection_pp':100*(new['f1']-old['f1']),
                          'alpha_changed':new['alpha']!=old['alpha'],'old_alpha':old['alpha'],'new_alpha':new['alpha'],
                          'old_threshold':old['threshold'],'new_threshold':new['threshold'],
                          'PA_f1_at_PA_selection_pct':100*ev['pa_at_pa_selection'][method]['f1'],
                          'PA_f1_at_no_PA_selection_pct':100*ev['pa_at_no_pa_selection'][method]['f1']})
    groups={};keys=('dataset','variant','configuration','window_sz','dim_feedforward','format','protocol','method','auc_name')
    for row in rows:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    means=[]
    for key,values in sorted(groups.items()):
        if len({v['seed'] for v in values})!=len(values):raise RuntimeError('Duplicate seed in group')
        row=dict(zip(keys,key));row['seeds']=len(values)
        for metric in ('f1_pct','precision_pct','recall_pct','auc_binary_pct','all_positive_f1_pct'):
            vals=[v[metric] for v in values if v[metric] is not None]
            row[metric+'_mean']=statistics.mean(vals) if vals else None
            row[metric+'_std']=statistics.stdev(vals) if len(vals)>1 else None
        means.append(row)
    index={(r['dataset'],r['variant'],r['configuration'],r['seed'],r['format'],r['protocol'],r['method']):r for r in rows}
    precision_pairs=[]
    for row in rows:
        if row['format']=='fp32':continue
        key=tuple(row[k] for k in ('dataset','variant','configuration','seed'))+('fp32',row['protocol'],row['method'])
        base=index.get(key)
        if base is None:continue
        result={k:row[k] for k in ('dataset','variant','configuration','seed','format','protocol','method','auc_name')}
        for metric in ('f1_pct','precision_pct','recall_pct','auc_binary_pct'):
            result[metric+'_delta_from_FP32_pp']=row[metric]-base[metric] if row[metric] is not None and base[metric] is not None else None
        precision_pairs.append(result)
    folder=root/'tables';folder.mkdir(parents=True,exist_ok=True)
    for name,values in [('per_seed',rows),('mean_std',means),('selection_comparison',pairs),('precision_comparison',precision_pairs)]:
        write_csv(folder/(name+'.csv'),values)
    return {'per_seed_rows':len(rows),'mean_std_rows':len(means),'selection_comparisons':len(pairs),'precision_comparisons':len(precision_pairs)}
