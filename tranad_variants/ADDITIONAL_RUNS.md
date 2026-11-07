# Estensione dello studio architetturale: 168 nuove run

Questa campagna completa le 72 run iniziali fino a 240 esperimenti:
3 dataset × 4 architetture × 4 configurazioni × 5 seed.
Non include i benchmark legacy a sei dataset.

| Dataset | reference W / FF | window_only W / FF | ff_only W / FF | compact W / FF |
|---|---:|---:|---:|---:|
| OPS-SAT | 10 / 8 | 5 / 8 | 10 / 4 | 5 / 4 |
| SMAP | 10 / 50 | 5 / 50 | 10 / 25 | 5 / 25 |
| MSL | 50 / 110 | 25 / 110 | 50 / 55 | 25 / 55 |

- `reference` e `compact`: aggiunge soltanto i seed 2029 e 2030, per 48 run.
- `window_only` e `ff_only`: esegue 2026, 2027, 2028, 2029, 2030, per 120 run.
- Modelli: `reference`, `shared_decoder`, `single_pass`, `encoder_only`.
- Ordine: tutte le 56 run OPS-SAT, poi le 56 SMAP, poi le 56 MSL.
- Ogni run esegue 50 epoche e subito dopo la propria evaluation.
- Dati, split, scaler, batch, LR, loss, scheduler, FP32, GPU cache e metriche
  conservano le implementazioni della prima campagna. Lo split OPS-SAT resta
  fisso con seed 2026; cambiano i seed del training.

## Trasferimento dal Mac

Dalla radice del progetto locale, con nessuna campagna architetturale in esecuzione:

```bash
rsync -avh --checksum --partial --progress \
  --exclude='__pycache__/' --exclude='*.pyc' \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  tranad_variants \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/
```

Il comando aggiorna soltanto il codice in `tranad_variants/`. Non trasferisce
risultati né modifica `tranad_plus/` o i dati. Non servono pacchetti aggiuntivi.
I tre dataset sono quelli già usati nella prima campagna.

## Avvio sul server

```bash
ssh -J atuscano@gate.pd.infn.it atuscano@lxa100
```

Poi, nella shell del server:

```bash
tmux new -s toast_architectures_extra
```

Dentro tmux:

```bash
cd /home/dliotta/AD-Satellite-Telemetry/tranad_variants
source ../ad-satellite-telemetry/bin/activate
bash run_additional.sh --device cuda:0
```

Lo script prima verifica sorgenti, dati e split delle 12 coppie dataset/configurazione.
Se la verifica passa, avvia le 168 run con training ed evaluation, una alla volta.
Il runner mantiene il comportamento originale sugli errori: registra il fallimento,
prosegue e restituisce codice finale 1 se una run è fallita.
Per fermarsi al primo errore aggiungere `--fail-fast`.

Per visualizzare soltanto il piano, senza creare risultati:

```bash
python run_experiments.py --campaign additional --dry-run
```

Per lasciare lavorare tmux: Ctrl+B, poi D. Per rientrare:

```bash
tmux attach -t toast_architectures_extra
```

## Risultati e ripresa

I risultati sono salvati nello stesso albero della prima campagna:

```text
results/v13/architectures/<dataset>/<modello>/<configurazione>/seed_<seed>/
```

Checkpoint ogni epoca, stato optimizer/RNG/scheduler, log, score, metriche e
riepiloghi sono conservati come nella prima campagna. Rilanciare lo stesso
`bash run_additional.sh --device cuda:0` per riprendere un'interruzione:
le nuove run complete vengono verificate e saltate, quelle interrotte riprese.
Con lo stesso ambiente di esecuzione, alla fine si attendono 240 run complete
e 48 gruppi con cinque seed ciascuno in `aggregate.json`.

Le vecchie 72 run non sono selezionate né riscritte dalla campagna `additional`.
Il piano storico `all` continua a descrivere le vecchie 72, non la griglia estesa.
L'aggiornamento della pianificazione cambia l'hash dei sorgenti dei nuovi
esperimenti; i vecchi manifest restano correttamente associati al codice storico.
Non rilanciare `--campaign all` per questa estensione e non forzare l'adozione di
checkpoint con una firma differente. Le implementazioni numeriche non cambiano.

Per trasferire i risultati aggiornati, dal Mac nella radice del progetto:

```bash
mkdir -p results/v13/architectures
rsync -avh --checksum --partial --progress \
  -e "ssh -J atuscano@gate.pd.infn.it" \
  atuscano@lxa100:/home/dliotta/AD-Satellite-Telemetry/results/v13/architectures/ \
  results/v13/architectures/
```
