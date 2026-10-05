# Progetto 1 — PointNet e fine-tuning sui cavalli

Progetto di **3D Perception, Learning-Based Data Fusion**, Università di Parma.
Il lavoro comprende un'applicazione per esplorare la classificazione di oggetti 3D
con PointNet e un esperimento separato che aggiunge la categoria **horse/cavallo**
tramite fine-tuning dell'ultimo livello.

**I dataset non sono inclusi nel repository.** Si scaricano dalle fonti originali
con gli strumenti descritti sotto. Git LFS non è necessario per questa modalità.
I pesi PointNet sono separati dai dataset: se presenti nella cartella `modello/`,
possono essere caricati nel repository; il downloader li recupera se mancanti.

## Percorso del progetto

1. Acquisizione e verifica di architettura, pesi preaddestrati e dataset ModelNet40.
2. Lettura delle point cloud, campionamento, centratura e normalizzazione.
3. Inferenza e osservazione di T-Net, feature e max pooling.
4. Interfaccia interattiva e verifica della coerenza delle visualizzazioni.
5. Esperimento finale: aggiunta di cavallo come 41ª classe e valutazione del
   compromesso tra nuova categoria e categorie originali.

L'app principale usa pesi già addestrati e non esegue training. Nell'esperimento
cavalli si aggiorna soltanto il classificatore finale.

## Struttura

| Percorso | Contenuto |
|---|---|
| `app.py` | Applicazione Streamlit per esplorare gli oggetti ModelNet40 |
| `inferenza.py` | Lettura dei dati, preprocessing, caricamento del modello e analisi |
| `modello/` | Architettura PointNet, configurazione, pesi e documentazione originale |
| `dati/` | Dataset ModelNet40 scaricato localmente e documentazione originale |
| `strumenti/download_resources.py` | Download e verifica delle risorse di base |
| `strumenti/test_inferenza.py` | Test del preprocessing e delle rappresentazioni interne |
| `esperimento_horse/scarica_cavalli.py` | Acquisizione dei cavalli e conversione in point cloud |
| `esperimento_horse/esperimento.py` | Fine-tuning e salvataggio delle misure |
| `esperimento_horse/visualizza.py` | Dashboard delle prove salvate |
| `esperimento_horse/risultati/` | Protocollo, metriche, predizioni e classificatori sperimentali |
| `provenienza.json` | Origine, revisioni e hash delle risorse acquisite |

## 1. Preparare l'ambiente

L'ambiente locale del progetto usa **Python 3.11.9**. I comandi seguenti sono per
PowerShell su Windows e vanno eseguiti dalla cartella principale del progetto.

Scaricare il repository con **Code → Download ZIP**, estrarlo e aprirlo in VS Code,
oppure clonarlo usando l'URL mostrato dal pulsante **Code** su GitHub.

Creare l'ambiente virtuale e installare le dipendenze:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Per eseguire anche l'esperimento cavalli:

```powershell
.\.venv\Scripts\python.exe -m pip install -r esperimento_horse/requirements.txt
```

Non è necessario attivare l'ambiente: i comandi usano direttamente il suo Python.
Il progetto funziona su CPU, senza richiedere una GPU. Se Windows segnala un errore
per un percorso troppo lungo, collocare il progetto in una cartella più breve,
ad esempio `C:\Progetti\PointNet`.

## 2. Scaricare ModelNet40 e le risorse PointNet

Fonti:

- Dataset: [jxie/modelnet40-2048](https://huggingface.co/datasets/jxie/modelnet40-2048).
- Modello: [DavidHanSZ/pointnet-modelnet40](https://huggingface.co/DavidHanSZ/pointnet-modelnet40).

Eseguire:

```powershell
.\.venv\Scripts\python.exe strumenti/download_resources.py
```

Lo script scarica i file mancanti, controlla dimensioni e hash e registra la
provenienza. Il sorgente locale del modello già adattato viene preservato quando
è presente il relativo record in `provenienza.json`.

Al termine devono essere presenti:

- `dati/train.parquet`: **9.840** oggetti.
- `dati/test.parquet`: **2.468** oggetti.
- `modello/pytorch_model.bin`: pesi della rete.

ModelNet40 contiene **12.308 oggetti CAD in 40 categorie**. Ogni campione è una
point cloud di **2.048 punti XYZ**, non un'immagine. La rete usa un sottoinsieme di
1.024 punti centrati e normalizzati. I due file Parquet occupano complessivamente
circa 283 MiB; serve una connessione Internet per acquisirli.

Il downloader legge la revisione corrente dei repository e la fissa per quella
esecuzione. Per confrontarsi esattamente con un'esecuzione precedente, consultare
revisioni e hash registrati in `provenienza.json` e nei risultati salvati.

## 3. Avviare l'applicazione principale

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Aprire **[http://localhost:8501](http://localhost:8501)**.

L'app permette di scegliere split e categoria, scorrere gli oggetti, ruotare la
point cloud e premere **Invia alla rete**. Mostra la predizione, le cinque
alternative principali e un approfondimento sulle trasformazioni interne.

Il pallino verde o rosso riguarda il singolo esempio selezionato, non tutta la
categoria. La softmax esprime punteggi relativi, non una garanzia di correttezza.
I colori dei contributi al max pooling contano i canali attivi in cui un punto
fornisce il massimo; non sono una segmentazione o una misura causale di importanza.

Lasciare aperto il terminale mentre si usa l'app. Per fermarla premere **Ctrl+C**.
I link localhost funzionano sul computer che esegue il servizio: pubblicare il
repository su GitHub non pubblica automaticamente le applicazioni online.

## 4. Scaricare i cavalli

Fonte: [Objaverse di AllenAI](https://huggingface.co/datasets/allenai/objaverse),
selezionando la categoria `horse` nelle annotazioni LVIS.

```powershell
.\.venv\Scripts\python.exe esperimento_horse/scarica_cavalli.py
```

Lo script scarica le mesh selezionate, esclude GLB oltre 12 MB e campiona 2.048
punti sulla superficie in proporzione all'area dei triangoli. Salva:

- `esperimento_horse/dati/*.npy`: coordinate XYZ dei cavalli.
- `esperimento_horse/dati/manifest.json`: ID, URL e hash dei file acquisiti.

Il manifest può essere incluso nel repository anche senza i file `.npy`.
Il downloader riprende i download considerando disponibili soltanto i campioni
presenti localmente con hash corretto. Il numero recuperato dipende dalle risorse
accessibili: nell'esecuzione documentata erano disponibili 55 cavalli nel pool di
training e 20 nel test, dopo deduplicazione.

Le annotazioni non sono state revisionate manualmente. Un nuovo download non
garantisce che tutte le risorse dell'esperimento originale siano ancora disponibili.

## 5. Eseguire il fine-tuning

Dopo aver scaricato entrambi i dataset:

```powershell
.\.venv\Scripts\python.exe esperimento_horse/esperimento.py
```

Il protocollo predefinito usa:

- PointNet congelato e feature a 256 dimensioni estratte prima dell'ultimo livello.
- Un nuovo classificatore a 41 classi, inizializzato copiando le 40 classi originali.
- Budget di 1, 5, 10, 25, 50 e 100 cavalli; quelli non disponibili vengono saltati.
- Replay fisso di 200 oggetti delle vecchie categorie.
- Test fisso con 400 oggetti ModelNet40 e 20 cavalli.
- Tre seed, 11, 22 e 33; 300 passi Adam per prova e learning rate 0,001.
- Loss pesata per metà sui cavalli e per metà sul replay.

Si aggiornano tutte le 41 righe del nuovo classificatore; il resto della rete resta
congelato. Ogni prova riparte dall'originale. Le nuove misure vengono salvate in
una nuova cartella dentro `esperimento_horse/risultati/`.

I checkpoint `head_*.pt` contengono solo l'ultimo livello e il riferimento all'hash
del modello base. Per riprodurre esattamente una prova servono gli stessi dati,
ID, checkpoint base e impostazioni registrati nei file del protocollo.

## 6. Avviare l'applicazione dei cavalli

Aprire un secondo terminale nella cartella principale:

```powershell
.\.venv\Scripts\python.exe -m streamlit run esperimento_horse/visualizza.py --server.port 8502
```

Aprire **[http://localhost:8502](http://localhost:8502)**.

La dashboard legge l'ultima esecuzione con misure salvate. Mostra le curve e le
predizioni sui singoli cavalli del test. Non esegue nuovo training. La vista del
singolo cavallo usa il primo seed; i grafici aggregano tutti i seed disponibili.

Per visualizzare i cavalli di una vecchia esecuzione servono i rispettivi `.npy`,
oltre a protocollo e predizioni. Il grafico `curva_accuracy.html` nella cartella
dei risultati è invece consultabile offline senza caricare la geometria.

## Risultati documentati

Nell'esecuzione `20261001_181025`, con 50 cavalli di training la recall media sui
cavalli è **95,00%**, la precision media **33,76%** e l'accuracy sulle vecchie classi
**71,33%**, contro il **78,00%** della baseline sullo stesso sottoinsieme di test.
L'acquisizione della nuova classe comporta quindi un aumento dei falsi positivi e
una perdita di accuratezza sulle vecchie categorie.

Le barre dei grafici sono deviazioni standard tra seed, non intervalli di
confidenza. Queste misure esplorative su oggetti CAD non dimostrano generalizzazione
ai cavalli reali. Per i dettagli consultare [il protocollo dell'esperimento](esperimento_horse/README.md)
e i file salvati nella cartella dei risultati.

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s strumenti -p test_inferenza.py -v
.\.venv\Scripts\python.exe -m unittest discover -s esperimento_horse -p test_esperimento.py -v
```

I test di inferenza richiedono modello e dataset ModelNet40. I test del protocollo
cavalli usano dati sintetici e non richiedono il download dei cavalli.

## Pubblicare il progetto senza dataset

Il `.gitignore` esclude ambiente virtuale, cache, log, file temporanei e i dataset
Parquet/NumPy. Conservare nel repository sorgenti, dipendenze, documentazione,
metadati di provenienza e risultati. Non eliminare i dataset locali: l'esclusione
da Git permette di continuare a usarli sul proprio computer.

Prima del primo commit controllare l'elenco con `git status`: non devono comparire
`.venv/`, i due `.parquet`, le point cloud `.npy` o i log. Se erano già stati
inseriti in commit precedenti, aggiungerli al `.gitignore` non li rimuove dalla
cronologia: occorre gestire separatamente quei file già tracciati.

## Fonti e licenze

- Modello e architettura: [DavidHanSZ/pointnet-modelnet40](https://huggingface.co/DavidHanSZ/pointnet-modelnet40).
  La documentazione originale, inclusa la licenza MIT dichiarata, è conservata in
  [modello/README_originale.md](modello/README_originale.md).
- Dataset: [jxie/modelnet40-2048](https://huggingface.co/datasets/jxie/modelnet40-2048).
  La documentazione scaricata è conservata in [dati/README_originale.md](dati/README_originale.md).
- Cavalli: [allenai/objaverse](https://huggingface.co/datasets/allenai/objaverse),
  con selezione tramite annotazioni LVIS. Consultare le condizioni della fonte e
  dei singoli asset per il loro utilizzo.

Le licenze delle risorse esterne restano distinte da quelle del codice del progetto.
