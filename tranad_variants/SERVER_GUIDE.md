# Avvio delle campagne architetturali sul server INFN

**Estensione delle 72 run già completate:** seguire
[ADDITIONAL_RUNS.md](ADDITIONAL_RUNS.md). La campagna `additional` aggiunge
168 run e porta lo studio a cinque seed e quattro configurazioni W/FF.
Le istruzioni seguenti descrivono la campagna iniziale.

La nuova cartella va accanto a `tranad_plus/`. I dati rimangono nella cartella
condivisa `data/`. I benchmark attualmente in esecuzione usano il loro codice;
il trasferimento seguente aggiunge soltanto `tranad_variants/`.

## 1. Trasferimento dal Mac

Dal terminale **locale**, nella radice del progetto:

```bash
rsync -avh --partial --progress \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  tranad_variants \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/
```

Servono i dataset MSL/SMAP già processati e
`/home/dliotta/AD-Satellite-Telemetry/data/raw/OPSSAT/segments.csv`.
Se manca OPS-SAT, dal Mac:

```bash
rsync -avh --partial --progress \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  data/raw/OPSSAT \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/data/raw/
```

## 2. Verifica sul server

```bash
cd /home/dliotta/AD-Satellite-Telemetry/tranad_variants
source ../ad-satellite-telemetry/bin/activate
python run_experiments.py --campaign pilot --dry-run
python run_experiments.py --campaign pilot --check
```

Il dry-run deve elencare 9 run. Il check controlla sorgenti, hash dei dati,
dimensioni e split, senza training né scrittura di risultati. Se segnala
`Baseline source differs`, fermarsi e confrontare il file indicato: non disabilitare
il controllo e non sovrascrivere la baseline mentre è usata da un training.

## 3. Primo pilot: 9 training con evaluation immediata

Aspettare che la GPU sia libera dai benchmark correnti. Aprire il tmux da una
shell del server esterna ad altri tmux:

```bash
tmux new -s toast_architectures
```

Dentro tmux:

```bash
cd /home/dliotta/AD-Satellite-Telemetry/tranad_variants
source ../ad-satellite-telemetry/bin/activate
python -u run_experiments.py --campaign pilot --device cuda:0
```

Ogni run: training di 50 epoche, checkpoint a ogni epoca, evaluation, salvataggi,
aggiornamento dei riepiloghi; poi parte la successiva. Un errore viene registrato
e la coda prosegue. Il codice di uscita finale è 1 se qualche run è fallita.
Non viene ridotto automaticamente il batch in caso di memoria insufficiente.

Per lasciare lavorare: **Ctrl+B, poi D**. Per rientrare:

```bash
tmux attach -t toast_architectures
```

## 4. Campagne successive

Dopo aver verificato i risultati del pilot, avviare separatamente:

```bash
python -u run_experiments.py --campaign main --device cuda:0
python -u run_experiments.py --campaign compact --device cuda:0
python -u run_experiments.py --campaign encoder_only --device cuda:0
```

`main` verifica e riutilizza le 9 run del pilot, aggiungendone 18. Le altre
campagne aggiungono rispettivamente 27 e 18 run: totale 72 uniche. La campagna
`all` contiene la stessa unione, ma il percorso progressivo consente di
controllare il pilot prima di impegnare tutta la GPU. Tre seed sono una prima
stima della variabilità, non una garanzia di significatività.

## 5. Ripresa e sola evaluation

Dopo un'interruzione, rilanciare lo stesso comando. Vengono riutilizzate le run
complete verificate e ripresi i checkpoint intermedi compatibili.

```bash
python -u run_experiments.py --campaign pilot --stage evaluate --device cuda:0
```

Per un solo dataset aggiungere `--dataset OPS-SAT` (oppure SMAP/MSL); per un solo
modello, per esempio, `--variant single_pass`. Un filtro non cambia gli
identificativi delle run, quindi i risultati restano riutilizzabili nelle campagne.

## 6. Risultati da conservare

Tutto vive in `../results/v13/architectures/`: cartelle delle singole run,
checkpoint, log, score, metriche, manifest, rapporti delle campagne,
`all_experiments.csv` e `aggregate.json`.

Per trasferire in locale l'intera fase, dal **Mac** nella radice del progetto:

```bash
rsync -avh --partial --progress \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/results/v13 \
  results/
```

POT e Best-F1 con point adjustment e AUC binaria come gli autori rimangono
distinti da BF denso, AUC degli score continui e selezione OPS-SAT su validation.
I valori di alpha dei modelli a singola uscita sono non applicabili.
