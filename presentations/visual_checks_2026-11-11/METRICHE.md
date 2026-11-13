# Come leggere i risultati senza confondere le domande

## Che cosa sappiamo dalle etichette

OPS-SAT: sappiamo se un intero frammento è nominale o anomalo. Non sappiamo quali singoli campioni costituiscano il difetto. Il codice assegna la stessa etichetta a tutti i suoi campioni per calcolare le metriche puntuali. Le discese a zero sono caratteristiche visibili, non annotazioni del loro preciso inizio e fine. Un falso negativo puntuale su OPS-SAT significa quindi «campione di un frammento positivo senza allarme», non necessariamente «guasto fisico mancato in quel campione».

SMAP/MSL: disponiamo di intervalli temporali annotati. Possiamo confrontare le decisioni con quelle annotazioni, senza assumere che le etichette spieghino la causa fisica o che ogni campione dell'intervallo presenti una deviazione visibile.

## Tre domande diverse

1. **F1 puntuale, senza PA:** quali singoli campioni vengono segnalati rispetto alle etichette puntuali? Su OPS-SAT le etichette sono estese dal frammento e non misurano accuratamente la localizzazione fisica.
2. **F1 dopo PA:** basta un allarme in un intervallo positivo per accreditare anche i suoi altri campioni. Gli allarmi nei tratti nominali restano singoli falsi positivi. È il protocollo storico di confronto, non il risultato direttamente prodotto dal modello.
3. **F1 per frammento, OPS-SAT:** ogni frammento produce una decisione e pesa uno. Anche il falso allarme in un frammento nominale conta a livello di frammento. È una valutazione diversa dal F1 dopo PA.

Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN). La formula non cambia: cambiano l'unità contata e le predizioni eventualmente modificate da PA. PA non modifica pesi, ricostruzioni o score e non elimina falsi positivi. Usa le etichette di test per estendere le predizioni positive negli intervalli annotati: non è una funzione disponibile in esercizio senza etichette.

## Esempio reale: perché gli stessi allarmi producono numeri molto diversi

OPS-SAT, TranAD+ reference, W10/FF8, seed2026, alpha fisso .5, soglia BF200 0.0013161241334898825. È un singolo controllo, non la media della campagna né il benchmark con alpha ottimizzato.

| Conteggio | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Campioni, senza PA | 2542 | 4855 | 26243 | 34.37% | 8.83% | 14.05% |
| Campioni, dopo PA | 26819 | 4855 | 1966 | 84.67% | 93.17% | 88.72% |
| Frammenti, un voto ciascuno | 106 | 392 | 7 | 21.29% | 93.81% | 34.70% |

La terza riga dichiara positivo un frammento se contiene almeno un allarme, con la stessa identica soglia puntuale. Non c'è una nuova ricerca della soglia. Rileva 106 dei 113 frammenti positivi, ma segnala anche 392 dei 416 nominali. Il numero alto dopo PA non dimostra dunque una buona classificazione dei frammenti. Viceversa, il F1 puntuale basso non dimostra da solo una cattiva localizzazione fisica su OPS-SAT, perché questa non è annotata.

Il massimo per frammento è sensibile ad allarmi isolati e durata del frammento. Non è l'unico aggregatore possibile e questa soglia non è stata ottimizzata per tale compito. Nella V20 esiste già una valutazione separata con 95° percentile per frammento e soglia selezionata sui frammenti di validation, poi fissata sul test. Sono protocolli distinti: non sostituire le loro metriche fra loro o scegliere retroattivamente quello più favorevole.

I conteggi completi delle quattro configurazioni, BF200/POT, sono in metric_units_opssat.csv; hash e regole sono in metric_units_provenance.json.

## La soglia e PA sono due operazioni separate

- BF200 sceglie fra 200 soglie usando le etichette di test. Nella valutazione storica ottimizza il F1 dopo PA e può anche selezionare alpha. È un confronto oracle, non una soglia pronta per uso onboard.
- POT stima una soglia attraverso SPOT. Nel protocollo storico la selezione di alpha usa comunque le etichette di test. L'implementazione concreta eredita inoltre il comportamento SPOT documentato nell'audit: non assumere che sia una quantile GPD invariabile determinata esclusivamente dal training.
- **Senza PA alla stessa soglia:** isola l'effetto dell'aggiustamento. Non mostra necessariamente il miglior F1 senza PA.
- **Senza PA con selezione dedicata:** cerca alpha/soglia per il F1 puntuale (nei limiti del metodo). La V18 contiene già questa valutazione. Non chiamarla uguale alla precedente.

Nel PDF attuale alpha è fisso .5; BF200 è scelto per PA, POT segue la procedura esistente; il confronto prima/dopo PA mantiene la stessa soglia. Tutti gli esempi usano seed2026, architettura TranAD+ reference e quattro coppie W/FF. Non confrontare direttamente questi controlli con medie di seed o benchmark con alpha scelto separatamente.

## Nomi definitivi per AUC

- **AUC-score:** ROC-AUC sui punteggi continui. Nessuna singola soglia, nessun PA. Può cambiare se cambia alpha o lo score; a score identici non cambia passando da BF a POT. Su OPS-SAT specificare anche se i punteggi sono per campione o per frammento.
- **AUC-bin:** ROC-AUC delle predizioni binarie prima di PA, equivalente a (recall + specificità)/2.
- **AUC-PA:** stessa quantità sulle predizioni dopo PA, come nel codice degli autori.

Non etichettare tutte e tre semplicemente «AUC» e non confrontarle come se fossero la stessa misura.

## Come presentare la tesi

**Confronto con gli autori:** conservare BF200/POT, F1/precision/recall dopo PA e AUC-PA, dichiarando selezione su test e regole particolari. I risultati restano quelli del protocollo, non una dimostrazione di prestazioni operative.

**Comprensione del comportamento:** leggere gli allarmi diretti, il tasso di falsi positivi e gli eventi rilevati insieme al F1 dopo PA. Per SMAP/MSL affiancare il F1 puntuale senza PA; per OPS-SAT affiancare le valutazioni per frammento già presenti nella V20. Non inventare etichette dei singoli difetti guardando le figure.

**Confronti fra modelli:** stesso dataset, stesso gruppo di configurazioni, stessi seed, precisione e regole di selezione/valutazione. Riportare medie e variabilità. Un miglioramento solo dopo PA va chiamato miglioramento del protocollo PA; una riduzione dei falsi allarmi diretti, a copertura comparabile, costituisce un'evidenza più concreta del comportamento utile.

Nessun nuovo training è necessario per chiarire queste distinzioni. I risultati storici e i checkpoint non sono stati modificati.
