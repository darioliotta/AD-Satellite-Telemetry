"""Readable tables for recovered OPS-SAT data; keep source families distinct."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,json,statistics as st
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/analysis_opssat_fragments'
VARIANTS=['reference','shared_decoder','single_pass','encoder_only']
NAMES=dict(zip(VARIANTS,['TranAD+','One-decoder','One-pass','Encoder-only']))
CONFIGS=['reference','window_only','ff_only','compact']

def csvwrite(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)

def metrics(r):
    a=r['metrics']['p95'];v=a['validation_BF200'];p=r['profile']
    return {'identity':r['identity'],'architecture':p['variant'],'configuration':p.get('configuration',''),
            'seed':p['seed'],'window':p['window_sz'],'ff':p['dim_feedforward'],'epochs':p['epochs'],
            'format':p.get('format','fp32'),'recovery_mode':r['recovery']['mode'],
            'alpha':r['recovery'].get('alpha'),
            **{k:v[k] for k in ('precision','recall','f1','TP','FP','TN','FN','threshold','AUC_bin')},
            'AUC_score':a['AUC_score'],'AP_score':a['AP_score'],
            'all_positive_F1':a['all_positive']['f1'],
            'validation_nominal':r['split_counts']['validation']['fragments']-r['split_counts']['validation']['anomalous'],
            'validation_anomalous':r['split_counts']['validation']['anomalous'],
            'test_nominal':r['split_counts']['test']['fragments']-r['split_counts']['test']['anomalous'],
            'test_anomalous':r['split_counts']['test']['anomalous']}

def average(group,**identity):
    assert group
    row={**identity,'evaluations':len(group),'seeds':len({r['seed'] for r in group})}
    for k in ('precision','recall','f1','AUC_score','AP_score','AUC_bin','all_positive_F1'):
        values=[100*r[k] for r in group];row[k+'_mean_pct']=st.mean(values)
        # Aggregate conditions within each seed before describing training-seed variability.
        byseed=defaultdict(list)
        for r in group:byseed[r['seed']].append(100*r[k])
        means=[st.mean(v) for v in byseed.values()]
        row[k+'_seed_SD_pp']=st.stdev(means) if len(means)>1 else None
    return row

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in r)+' |' for r in rows])

def main():
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('*.json'))]
    families=defaultdict(list);mainrows=[]
    for r in cases:
        ident=r['identity'];family=None
        if ident.startswith(('legacy:results/v10/','legacy:results/v12/')):family='ablation_grid'
        elif ident.startswith('archive:results/v13/'):family='architecture'
        elif ident.startswith('quant:results/v16/'):family='weights_only'
        elif ident.startswith('quant:results/v17/'):family='integer_CPU_reference'
        elif ident.startswith('v20:results/v20/mechanisms/analysis/OPS-SAT/'):
            family='integer_A100'
        if family and r['primary_available']:
            row={'family':family,**metrics(r)};families[family].append(row);mainrows.append(row)
    csvwrite('main_studies_per_model.csv',mainrows)
    numerical=[]
    for r in cases:
        if not r['identity'].startswith('quant:'):continue
        p=r['profile'];audit=r['recovery']['historical_numerical_checks']
        for method,check in audit['old_PA_endpoints'].items():
            numerical.append({'identity':r['identity'],'kind':r['recovery']['quantization_kind'],
                'architecture':p['variant'],'configuration':p['configuration'],'seed':p['seed'],'format':p['format'],
                'historical_method':method,'exact_historical_PA_counts':check['exact_match'],
                **{k+'_delta':v for k,v in check['historical_confusion_delta'].items()}})
    csvwrite('CPU_replay_numerical_audit.csv',numerical)
    groups=defaultdict(list)
    for r in families['ablation_grid']:groups[r['window'],r['ff']].append(r)
    ablation=[average(v,window=w,ff=h) for (w,h),v in sorted(groups.items())]
    for a in ablation:assert a['seeds']==3 and a['evaluations']==3
    assert len(ablation)==30
    csvwrite('ablation_30_cells.csv',ablation)
    arch=[]
    for variant in VARIANTS:
        for config in CONFIGS:
            group=[r for r in families['architecture'] if r['architecture']==variant and r['configuration']==config]
            assert len(group)==5
            arch.append(average(group,architecture=variant,configuration=config))
    csvwrite('architecture_16_cells.csv',arch)
    # Preserve the original, distinct validation-selected fragment ranking too.
    # These reports did not save segment-level confusion counts; never relabel
    # their POINTWISE precision/recall as fragment-level performance.
    selected=[]
    for root in (ROOT/'results/v13/architectures/OPS-SAT',ROOT/'results/v14/continuation/OPS-SAT'):
        for path in sorted(root.glob('**/metrics.json')):
            original=json.loads(path.read_text());value=original.get('validation_selected',{})
            if value.get('status')!='measured':continue
            p=original['profile'];chosen=value['selection']['chosen']
            selected.append({'source':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'architecture':p['variant'],'configuration':p['configuration'],'seed':p['seed'],'epochs':p['epochs'],
                'alpha':chosen.get('alpha'),'calibration':chosen['calibration'],'aggregation':chosen['aggregation'],
                'validation_AUC_score':chosen['validation_fragment_auc'],
                'test_AUC_score':value['fragment']['roc_auc'],'test_AP_score':value['fragment']['average_precision'],
                'segment_F1':'not_saved_in_original_report'})
    csvwrite('original_validation_selected_fragment_auc.csv',selected)
    quant=[]
    # Preserve A100-only 3-seed integer comparisons. Extra CPU seeds stay explicit.
    for family in ('weights_only','integer_A100','integer_CPU_reference'):
        for fmt in ('fp32','int8','int4'):
            if family=='weights_only' and fmt=='fp32':group=families['architecture']
            elif family=='integer_CPU_reference' and fmt=='fp32':
                seeds={r['seed'] for r in families[family]}
                group=[r for r in families['architecture'] if r['seed'] in seeds]
            else:group=[r for r in families[family] if r['format']==fmt]
            if group:quant.append(average(group,family=family,format=fmt))
    csvwrite('quantization_summary.csv',quant)
    # Main coverage: 90 grid models, 80 architecture models, 160 weights-only,
    # 96 original A100 integer + 64 CPU recovered integer (the last kept separate).
    expected={'ablation_grid':90,'architecture':80,'weights_only':160,'integer_A100':144,'integer_CPU_reference':64}
    actual={k:len(families[k]) for k in expected}
    coverage_ok=actual==expected
    report=['# OPS-SAT: ricostruzione delle valutazioni per frammento','',
       '**Nessun training aggiunto. Gli esperimenti originali restano invariati.**','',
       'Protocollo comune: uno score p95 per frammento, alpha fisso .5 nei modelli a due uscite (nessun alpha negli altri), soglia BF200 scelta sui frammenti originali di validation, poi fissata sul test. Ogni frammento pesa uno; nessun PA. Le metriche storiche BF200/POT puntuali con PA restano negli archivi originali, come protocollo distinto.','',
       '## Copertura degli studi principali','',table(['Famiglia','Attesi','Recuperati'],[[k,expected[k],actual[k]] for k in expected]),'',
       'La riga integer_A100 include 48 FP32 + 48 INT8 + 48 INT4 originali V20, seed2026–2028. I 64 casi integer_CPU_reference sono INT8/INT4 dei seed2029–2030: nuova sola evaluation su CPU con accumulazione INT32, non misure GPU né risultati dichiarati identici alla A100. Sono tenuti separati nelle medie.','',
       '## Ablation W / FF','',
       'Tutte le 30 celle × 3 seed sono in ablation_30_cells.csv. Esempi di confronti; percentuali, media di tre seed:', '',
       table(['W','FF','Precision','Recall','F1','AUC-score'],[[a['window'],a['ff'],*[f"{a[k+'_mean_pct']:.2f}" for k in ('precision','recall','f1','AUC_score')]] for a in ablation if (a['window'],a['ff']) in [(10,8),(5,8),(10,4),(5,4),(3,8),(10,2),(3,2)]]),'',
       '## Architetture: W10 / FF8','',
       'Media di cinque seed, FP32. AUC-score usa gli score continui p95 per frammento.','',
       table(['Modello','Precision','Recall','F1','AUC-score'],[[NAMES[a['architecture']],*[f"{a[k+'_mean_pct']:.2f}" for k in ('precision','recall','f1','AUC_score')]] for a in arch if a['configuration']=='reference']),'',
       '## Quantizzazione','',
       'Medie sulle 4 architetture × 4 configurazioni W/FF per seed. La variabilità riportata nei CSV è calcolata fra le medie dei seed, non trattando ogni configurazione come seed indipendente.','',
       table(['Famiglia','Formato','Seed','Precision','Recall','F1','AUC-score'],[[a['family'],a['format'],a['seeds'],*[f"{a[k+'_mean_pct']:.2f}" for k in ('precision','recall','f1','AUC_score')]] for a in quant]),'',
       '## Interpretazione da non perdere','',
       'Nel test ci sono 113 frammenti positivi e 416 nominali. Segnalare tutti i frammenti produce precision 21.36%, recall 100% e F1 35.20%. Valori vicini a questa terna non indicano un buon rilevatore.','',
       'La validation originale contiene 321 frammenti anomali e 254 nominali (55.83% positivi); il test 113/529 (21.36%). La soglia che massimizza F1 sulla validation puo quindi essere troppo permissiva sul test. Questa differenza e un limite della calibrazione, non una nuova contaminazione del training. Anche AUC-score e AP-score vanno letti: non dipendono dalla singola soglia e non garantiscono buoni valori.','',
       'Il percentile95 e una regola fissa di aggregazione, non la definizione del dataset. Puo attenuare anomalie che occupano pochi campioni; il massimo e sensibile a singoli falsi allarmi e alla lunghezza dei frammenti. Dove sono disponibili entrambi sono riportati separatamente in all_fragment_metrics.csv, senza scegliere quello piu favorevole sul test.','',
       'Le valutazioni per frammento gia presenti in V13/V14, che selezionavano alpha, calibrazione e aggregazione sulla validation, sono conservate in original_validation_selected_fragment_auc.csv. Sono un protocollo diverso dal controllo p95/alpha=.5: non sono state cancellate o sostituite dai nuovi valori. Riportavano AUC/AP per frammento, ma non precision/recall/F1 per frammento; le metriche puntuali presenti accanto non sono state scambiate per metriche dei segmenti.','',
       '## Verifica numerica della ricostruzione locale','',
       'Tutti gli export quantizzati sono verificati tramite hash. Le statistiche degli score ricalcolati su CPU rispettano il controllo storico rtol=5e-4, atol=1e-7. Questo non implica identiche decisioni alla soglia su piattaforme diverse. Le differenze ai vecchi endpoint BF/POT sono registrate in CPU_replay_numerical_audit.csv.','',
       'Un caso richiede particolare attenzione: weight-only INT8, one-pass/compact, seed2026, vecchia soglia POT 0.003885347260902603. Nella CPU replay si osservano 7838 FP in piu rispetto al conteggio A100 storico. Lo score storico al percentile90 e 0.003885346930474043, vicinissimo alla soglia: confronti stretti su molti valori quasi uguali possono essere sensibili ad arrotondamenti. Questa e una possibile spiegazione, non una dimostrazione della posizione di ogni score GPU, che non e stato archiviato. I vecchi numeri POT non sono sovrascritti; riprodurre esattamente quelle decisioni richiede sola evaluation sulla A100. Il nuovo endpoint per frammento usa invece la propria soglia di validation, chiaramente distinta.','',
       'I risultati CPU sono quindi nuovi risultati di evaluation da pesi originali, non una dichiarazione di equivalenza bit per bit delle campagne GPU. La copertura completa significa disponibilita dei dati per gli studi, non equivalenza numerica perfetta fra tutte le piattaforme.','',
       '**Non emerge automaticamente un calo monotono delle prestazioni quando il modello viene semplificato.** Il passaggio ai frammenti ridimensiona le vecchie metriche PA; non giustifica una conclusione opposta senza guardare i dati.','',
       '## Archivi aggiuntivi','',
       'Sono recuperati anche i dati compatibili di V6/V8/V9, continuation V14, controlli del contesto V14/V15, training di controllo V15 e baseline/controlli V20. Restano separati: non aumentano artificialmente il numero di seed degli studi principali. I casi senza validation storica indipendente sono identificati e non ricevono una soglia fittizia.','',
       'La V19 riguarda la regolarizzazione: le sue checkpoint e metriche scalari sono inventariate, ma il recupero puntuale dei suoi score non e necessario per ablation, architetture o quantizzazioni. Le vecchie cartelle legacy non contengono esperimenti OPS-SAT sotto quel nome. Non esiste results/v7 nella copia locale.','',
       '## File e riproducibilita','',
       '- main_studies_per_model.csv: ogni modello/formato con provenienza e conteggi esatti.','- all_fragment_metrics.csv: tutti i protocolli recuperati, inclusi oracoli di test espliciti e diagnostiche max/p95.','- cases/: piccoli array per frammento, metriche, hash e controlli numerici; niente archivi giganti di ricostruzioni.','- inventory.csv: inventario degli archivi, compresi quelli senza dati direttamente aggregabili.','- coverage.json: stato del recupero e fallimenti ancora da risolvere.','',
       'Sola evaluation locale, da rilanciare in modo riprendibile:','',
       '```bash','python -B tranad_plus/reports/recover_opssat_fragments.py --phase all','python -B tranad_plus/reports/summarize_opssat_fragments.py','```','']
    (OUT/'REPORT.md').write_text('\n'.join(report))
    writebook(mainrows,ablation,arch,quant)
    (OUT/'main_coverage.json').write_text(json.dumps({'expected':expected,'actual':actual,'complete':coverage_ok,
        'training_added':0,'integer_missing_seeds_CPU_only':True},indent=2)+'\n')
    print(json.dumps({'expected':expected,'actual':actual,'complete':coverage_ok},indent=2))

def writebook(mainrows,ablation,arch,quant):
    try:from openpyxl import Workbook
    except ImportError:return
    from openpyxl.styles import Font,PatternFill,Alignment
    wb=Workbook();wb.remove(wb.active)
    extra=[]
    for name,file in [('All_recovered','all_fragment_metrics.csv'),('Original_selected_AUC','original_validation_selected_fragment_auc.csv'),('Inventory','inventory.csv')]:
        if (OUT/file).exists():
            with (OUT/file).open() as f:extra.append((name,list(csv.DictReader(f))))
    for name,rows in [('Models',mainrows),('Ablation',ablation),('Architectures',arch),('Quantization',quant)]+extra:
        if not rows:continue
        ws=wb.create_sheet(name);keys=list(rows[0]);ws.append(keys)
        def cell_value(value):
            if isinstance(value,str):
                if value=='':return None
                try:return float(value)
                except ValueError:return value
            return value
        for r in rows:ws.append([cell_value(r.get(k)) for k in keys])
        ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
        for cell in ws[1]:cell.fill=PatternFill('solid',fgColor='9B0014');cell.font=Font(color='FFFFFF',bold=True)
        for col in ws.columns:
            col=list(col);width=min(45,max(12,len(str(col[0].value))+2));ws.column_dimensions[col[0].column_letter].width=width
            for cell in col[1:]:
                if isinstance(cell.value,float):cell.number_format='0.0000'
    notes=wb.create_sheet('Read_me');notes.append(['Read REPORT.md for protocol, aggregation, platform and baseline limitations.'])
    notes.append(['Models stores fractions 0–1; summary sheets have explicitly named percentages.'])
    notes.append(['No training added. Original archives unchanged. CPU reference integer evaluations are separate from A100.'])
    wb.save(OUT/'OPS_SAT_recovered.xlsx')

if __name__=='__main__':main()
