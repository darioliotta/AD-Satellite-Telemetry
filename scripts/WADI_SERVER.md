# Training WADI su un nuovo server

Il pacchetto `WADI_original_training.tar.gz` contiene il codice corrente e il
launcher, con una struttura indipendente dal percorso del progetto originale.
I dati, gli ambienti Python e i risultati precedenti non sono inclusi.

Le configurazioni sono quelle del catalogo di replica:

| Selezione | Window | FF | Batch | Learning rate | Epoche |
|---|---:|---|---:|---:|---:|
| `ws20` | 20 | `2x` | 8192 | 0.001 | 50 |
| `ws50` | 50 | `2x` | 2048 | 0.0001 | 50 |

Con i dati attuali `2x` corrisponde a 256. Si usano Adam standard, FP32 faithful,
padding e ordine dei batch originali. La cache del dataset in GPU è disabilitata.
Il launcher esegue solo training. La valutazione è un passaggio successivo.

## Trasferimento

Negli esempi sostituire `ALIAS_SSH` con l'alias/indirizzo SSH reale. Se l'accesso
richiede un jump host, usare l'alias già configurato per quel collegamento.

Dal computer locale, nella radice di AD-Satellite-Telemetry:

```bash
scp results/wadi_transfer/WADI_original_training.tar.gz ALIAS_SSH:~/
```

Sul server:

```bash
tar -xzf WADI_original_training.tar.gz
cd wadi_original_training
```

L'archivio crea anche le directory di destinazione vuote per i dati. Da un nuovo
terminale LOCALE, sempre dalla radice del progetto:

```bash
rsync -avPz data/processed/tranad_plus/WADI/Train/ ALIAS_SSH:~/wadi_original_training/data/processed/tranad_plus/WADI/Train/
rsync -avPz data/processed/tranad_plus/WADI/Test/ ALIAS_SSH:~/wadi_original_training/data/processed/tranad_plus/WADI/Test/
```

Per il solo training sono necessari i file in `Train/Tracks/`. Gli altri file
servono per la successiva valutazione. La cartella `Both/` non serve.
Si possono usare dati già presenti sul server impostando `TRANADPLUS_DATA_ROOT`
al percorso assoluto della cartella `data` che contiene `processed/tranad_plus`.

## Ambiente e GPU

Attivare l'ambiente Python CUDA del server. Le dipendenze del progetto sono in
`tranad_plus/requirements.txt`; per un nuovo ambiente l'installazione di PyTorch
va scelta in base al driver/GPU del server. Non copiare la virtualenv del Mac.

Prima dell'avvio:

```bash
python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available())'
nvidia-smi
bash scripts/train_wadi_original.sh all --dry-run
```

Su un cluster con scheduler, ottenere prima l'allocazione GPU secondo le regole
del centro ed eseguire questi comandi nel nodo assegnato. I comandi dello
scheduler dipendono dal server: questo pacchetto non assume Slurm/PBS specifici.
I precedenti tentativi su GPU da circa 15 GiB sono terminati per memoria esaurita;
la batch originale richiede un'allocazione con memoria sufficiente. Il launcher
non riduce automaticamente la batch.

## Avvio e ripresa

Dentro l'allocazione GPU e con l'ambiente Python attivato:

```bash
bash scripts/train_wadi_original.sh ws20
bash scripts/train_wadi_original.sh ws50
```

Oppure `bash scripts/train_wadi_original.sh all` per eseguirli in sequenza.
Il primo errore interrompe la sequenza. I checkpoint di recupero sono salvati
ogni epoca: rilanciare la singola configurazione interrotta per riprenderla.
Un checkpoint finale esistente viene protetto dall'overwrite; per proseguire
con l'altra configurazione indicare esplicitamente `ws20` oppure `ws50`.

I risultati sono in `results/wadi_original/v1/WADI/` dentro la cartella estratta.
`WADI_DEVICE=cuda:1` seleziona un'altra GPU visibile; rispettare sempre
`CUDA_VISIBLE_DEVICES` impostato dal sistema di allocazione.

## Verifica dei file trasferiti

`transfer_manifest.json` contiene gli SHA-256 del codice e dei quattro file dati
necessari per training e valutazione, calcolati sul computer di origine.
Per verificarli sul server, dalla radice estratta:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('transfer_manifest.json').read_text())
for entry in manifest['code'] + manifest['data']:
    p = Path(entry['path'])
    if not p.is_file():
        raise SystemExit(f'Missing: {p}')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != entry['sha256']:
        raise SystemExit(f'Checksum mismatch: {p}')
print('Code and data checksums match.')
PY
```
