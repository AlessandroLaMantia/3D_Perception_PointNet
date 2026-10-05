# Esperimento separato: imparare horse con pochi esempi

L'app esistente, i suoi sorgenti e i pesi originali non vengono modificati. Questa cartella contiene un esperimento indipendente, con una 41a classe: **horse/cavallo**, assente da ModelNet40.

## Esecuzione dal terminale della cartella principale

```powershell
.\.venv\Scripts\python.exe -m pip install -r esperimento_horse/requirements.txt
.\.venv\Scripts\python.exe esperimento_horse/scarica_cavalli.py
.\.venv\Scripts\python.exe esperimento_horse/esperimento.py
```

Aprire http://localhost:8502 per vedere curva, misure e singoli cavalli del test con la predizione di ogni prova.

## Dataset e limite della prova a 100

Fonte: [Objaverse su Hugging Face](https://huggingface.co/datasets/allenai/objaverse), selezione tramite `lvis-annotations.json.gz`, categoria `horse` (90 ID alla verifica iniziale). Il downloader legge solo quei modelli, scarta GLB maggiori di 12 MB, campiona 2.048 punti uniformemente rispetto all'area della superficie e conserva solo gli XYZ in `.npy`.

cavalli di training: 55
cavalli di test: 20


## Protocollo

1. Deduplicazione per hash della mesh, ordinamento stabile e split deterministico: 20 cavalli fissi di test, gli altri nel pool di training.
2. Replay fisso: 5 oggetti di training per ciascuna vecchia classe (200 totali). Il numero sull'asse X conta **solo i cavalli aggiuntivi**, non tutti gli esempi di training.
3. Test fisso: 10 oggetti per ciascuna vecchia classe (400) e i 20 cavalli. Le prove vengono confrontate sugli stessi oggetti.
4. Backbone e livelli fino al penultimo completamente congelati, BatchNorm in eval, dropout disattivato. Si estraggono una volta le feature a 256 dimensioni prima di `fc3`.
5. Sostituzione di `fc3` (256 -> 40) con un nuovo Linear (256 -> 41), copiando le 40 righe originali e inizializzando la nuova riga con seed controllato.
6. **Fine-tuning solo dell'ultimo livello**: Adam, learning rate 0,001, 300 passi per ogni prova. La loss pesa 50% i cavalli e 50% il replay. Il budget di ottimizzazione e uguale per tutti i numeri di esempi.
7. Tre seed (11, 22, 33) cambiano supporto e inizializzazione. Per ciascun seed i sottoinsiemi sono annidati: l'esempio della prova 1 e incluso nei 5, poi nei 10, ecc. Ogni prova riparte dall'originale, non dai pesi del budget precedente.
8. Nessuna selezione di epoche o iperparametri in base al test. I risultati sono una prima misura esplorativa, non una ricerca dell'assetto ottimale.

Questa e una forma limitata di fine-tuning supervisionato (ultimo livello), non un riaddestramento completo. Le feature congelate consentono un esperimento veloce anche su CPU. La domanda misurata e: **quanto aiuta aumentare i cavalli annotati, a parita di algoritmo e numero di passi?** Non determina da sola il numero di epoche necessario o le prestazioni di un fine-tuning completo.

## Metriche da leggere insieme

- **Recall horse**: cavalli classificati correttamente / cavalli nel test. E l'accuracy sul solo sottoinsieme cavalli, non sull'intero test.
- **Precision horse**: fra tutte le predizioni cavallo, quante sono davvero cavalli (0 quando non ci sono predizioni cavallo).
- **Accuracy vecchie classi**: misura l'eventuale perdita di capacita sulle categorie originali.
- **Accuracy complessiva**: corrette su tutti gli oggetti di test; e influenzata dalla presenza di 400 vecchi oggetti contro 20 cavalli.
- **Macro accuracy**: media delle accuracy delle 41 classi, dando lo stesso peso a ogni classe.
- **Old false horse rate**: fra gli oggetti delle vecchie classi, quanti vengono scambiati per cavalli.

Rispondere sempre horse darebbe recall 100% ma distruggerebbe accuracy sulle vecchie classi e precision. Per questo il grafico mostra piu curve. Le barre sono deviazioni standard tra seed, **non intervalli di confidenza**; il test resta identico. Con 20 cavalli la recall di ogni prova cambia a passi di 5 punti percentuali. Le curve possono scendere: un andamento crescente non e garantito.

Il protocollo salva anche la baseline originale a 40 classi, che per costruzione non puo predire horse. Gli esempi originali di test possono essere stati usati dall'autore per scegliere il checkpoint: la baseline sulle vecchie classi non e quindi un benchmark completamente nuovo. Inoltre horse viene da Objaverse e le altre classi da ModelNet40: differenze di dominio possono influenzare la discriminazione. Non dimostra generalizzazione ai cavalli reali o ad altri animali mai visti.

## Risultati salvati

- `protocollo.json`: impostazioni, split, indici del replay/test, hash originale, budget saltati e baseline.
- `supporti.json`: ID esatti dei cavalli usati per ciascuna prova.
- `misure.csv`: metriche e tempo del fine-tuning per budget e seed (esclusa estrazione feature).
- `riepilogo.json`: medie e deviazioni standard.
- `predizioni_*.npz`: predizioni ed etichette di test.
- `head_*.pt`: solo l'ultimo livello aggiornato e l'hash del modello base; non una copia completa di PointNet.
- `curva_accuracy.html`: grafico offline dei risultati realmente misurati.

Per ricostruire un modello sperimentale: caricare `carica_modello()`, sostituire `model.fc3` con `torch.nn.Linear(256, 41)` e caricare in questo livello `head_state_dict` dal checkpoint con `torch.load(..., weights_only=True)`. La classe 40 e horse. Usare `model.eval()` e lo stesso preprocessing; l'app originale resta a 40 classi.

## Verifiche

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s esperimento_horse -p test_esperimento.py -v
```

I test controllano split/deduplicazione, prova con un solo esempio, ripetibilita, conservazione dei pesi iniziali e metriche che penalizzano la risposta costante horse.
