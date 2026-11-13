"""Paired INT8/INT4-vs-FP32 and seed mean/SD, without pointwise archives."""
import json
from pathlib import Path
import statistics
from tranad_plus.core.common import hash_file
from tranad_plus.campaigns.temporal.summarize import write_csv


def summarize(root):
    root = Path(root)
    rows, pairs = [], []
    for path in sorted(root.glob('*/*/*/seed_*/status.json')):
        status = json.loads(path.read_text())
        if status['status'] != 'complete':
            continue
        if hash_file(path.parent/'metrics.json') != status['artifacts']['metrics.json']:
            raise RuntimeError('Changed metrics '+str(path))
        report = json.loads((path.parent/'metrics.json').read_text())
        p = report['profile']
        for name, result in report['formats'].items():
            for protocol in ('oracle', 'parent_locked'):
                for method in ('bf', 'pot'):
                    endpoint = result[protocol].get(method)
                    if not endpoint:
                        continue
                    row = {k:p[k] for k in ('dataset','variant','configuration','seed','window_sz','dim_feedforward')}
                    row.update(format=name,protocol=protocol,method=method,alpha=endpoint['alpha'],
                               threshold=endpoint.get(method+'_threshold'),execution=result['storage']['execution'])
                    for metric, suffix in [('f1_pct','f1'),('precision_pct','precision'),('recall_pct','recall'),('AUC_PA_pct','ROC/AUC')]:
                        value = endpoint.get(method+'_'+suffix)
                        row[metric] = 100*value if value is not None else None
                    value = endpoint.get('raw_without_PA',{}).get(method+'_f1')
                    row['raw_f1_pct'] = 100*value if value is not None else None
                    resource, storage = result['resources'], result['storage']
                    row.update(tensor_bytes=storage['tensor_bytes_including_bias_scales_buffers'],
                               resident_buffer_bytes=resource['buffer_bytes_resident'],
                               median_ms=resource['median_ms'],p95_ms=resource['p95_ms'],
                               analytic_MACs=resource['analytic_operations']['matmul_MACs_per_window'],
                               cuda_peak_allocated_bytes=resource['process_after'].get('cuda_peak_allocated_bytes'))
                    rows.append(row)
    keys = ('dataset','variant','configuration','seed','protocol','method')
    reference = {tuple(r[k] for k in keys):r for r in rows if r['format']=='fp32'}
    for row in rows:
        if row['format']=='fp32':continue
        baseline = reference[tuple(row[k] for k in keys)]
        pair = dict(row)
        for metric in ('f1_pct','precision_pct','recall_pct','AUC_PA_pct','raw_f1_pct'):
            pair[metric+'_delta_pp'] = row[metric]-baseline[metric] if row[metric] is not None and baseline[metric] is not None else None
        pair.update(tensor_compression_ratio=baseline['tensor_bytes']/row['tensor_bytes'],
                    measured_latency_ratio_fp32_over_integer=baseline['median_ms']/row['median_ms'])
        pairs.append(pair)
    grouped = {}
    keys = ('dataset','variant','configuration','window_sz','dim_feedforward','format','protocol','method','execution')
    for row in rows:grouped.setdefault(tuple(row[k] for k in keys),[]).append(row)
    means=[]
    for key, values in sorted(grouped.items()):
        if len({v['seed'] for v in values}) != len(values):raise RuntimeError('Duplicate seed')
        row=dict(zip(keys,key));row['seeds']=len(values)
        for metric in ('f1_pct','precision_pct','recall_pct','AUC_PA_pct','raw_f1_pct','tensor_bytes','median_ms','p95_ms'):
            numbers=[v[metric] for v in values if v[metric] is not None]
            row[metric+'_available_seeds']=len(numbers)
            row[metric+'_mean']=statistics.mean(numbers) if numbers else None
            row[metric+'_std']=statistics.stdev(numbers) if len(numbers)>1 else None
        means.append(row)
    folder=root/'tables';folder.mkdir(parents=True,exist_ok=True)
    for name, values in [('per_seed',rows),('paired_deltas',pairs),('mean_std',means)]:
        write_csv(folder/(name+'.csv'),values)
    return {'per_seed_rows':len(rows),'paired_rows':len(pairs),'mean_std_rows':len(means)}
