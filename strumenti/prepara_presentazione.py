r"""Genera slide HTML offline e testo per l'esposizione dai sorgenti locali.

Eseguire dalla radice: .\.venv\Scripts\python.exe strumenti/prepara_presentazione.py
Non esegue training: legge codice, metadati e risultati già salvati.
"""
import ast
import csv
import html
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'presentazione'


def estratto(file, function, start=None, end=None):
    # L'AST individua la funzione reale; le porzioni selezionate restano codice del progetto.
    source = (ROOT / file).read_text(encoding='utf-8')
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    lines = source.splitlines()[node.lineno - 1:node.end_lineno]
    if start:
        lines = lines[next(i for i, line in enumerate(lines) if start in line):]
    if end:
        lines = lines[:next(i for i, line in enumerate(lines) if end in line) + 1]
    return '\n'.join(lines)


def main():
    # L'esecuzione documentata viene fissata esplicitamente per rendere le slide riproducibili.
    OUT.mkdir(exist_ok=True)
    run = ROOT / 'esperimento_horse/risultati/20261001_181025'
    protocol = json.loads((run / 'protocollo.json').read_text(encoding='utf-8'))
    with (run / 'misure.csv').open(encoding='utf-8') as handle:
        measurements = list(csv.DictReader(handle))
    slides = []

    def add(title, bullets, notes, code='', source='', extra=''):
        # Ogni slide separa i concetti essenziali dalle note per la spiegazione orale.
        slides.append(dict(title=title, bullets=bullets, notes=notes, code=code, source=source, extra=extra))

    add('Progetto 1 · Capire PointNet attraverso il codice',
        ['Classificazione di oggetti 3D rappresentati come point cloud.',
         'Dai dati ModelNet40 alla visualizzazione dei passaggi interni.',
         'Estensione finale: apprendere la nuova classe cavallo con pochi esempi.'],
        'La presentazione ricostruisce il lavoro nell’ordine logico del progetto e si basa sui file presenti. Prima abbiamo predisposto dati e rete preaddestrata, poi il preprocessing e l’inferenza, infine la visualizzazione e un esperimento separato di fine-tuning. Non viene inventata una cronologia di attività non documentate.')
    add('1 · Obiettivo e percorso del lavoro',
        ['Acquisire e verificare modello e dataset.', 'Preparare gli XYZ e classificare un oggetto.',
         'Osservare trasformazioni, feature e max pooling.', 'Costruire l’interfaccia e verificare i calcoli.',
         'Solo alla fine: aggiungere horse e confrontare i risultati.'],
        'L’app di base non addestra la rete: usa pesi già appresi dall’autore. Il contributo locale consiste nel rendere il modello utilizzabile, esplorabile e verificabile; l’apprendimento locale compare nell’esperimento horse.')
    add('2 · Come è organizzato il progetto',
        ['modello/: architettura PointNet, configurazione e pesi.', 'dati/: train.parquet e test.parquet.',
         'inferenza.py: lettura, preprocessing, caricamento e analisi.', 'app.py: interfaccia Streamlit e grafici Plotly.',
         'strumenti/: download, test e generazione di questa presentazione.',
         'esperimento_horse/: dati, training della testa, dashboard e risultati.'],
        'Questa è la struttura effettivamente presente. Relazione e README citano anche esplora.py, avvia.ps1 ed esempio.ply, che non sono presenti nella cartella corrente: non li presentiamo come strumenti disponibili.')
    add('3 · Risorse di partenza e provenienza',
        ['Modello: DavidHanSZ/pointnet-modelnet40 su Hugging Face.',
         'Dataset: jxie/modelnet40-2048; 40 categorie di oggetti CAD.',
         'Ogni oggetto è una matrice di 2.048 punti con tre coordinate XYZ.',
         'provenienza.json registra URL, revisione, dimensione e SHA256.'],
        'Un punto contiene solo posizione, non colore o normali. Si tratta di oggetti CAD campionati, non di scene LiDAR. La separazione train/test è già fornita dal dataset. Le fonti sono registrate localmente: non stiamo verificando online nuove versioni delle risorse.')
    add('4 · Download verificato',
        ['La revisione del repository viene fissata prima del download.',
         'Un file .part evita di scambiare un download incompleto per un file valido.',
         'Dimensione e hash controllano l’integrità dei binari.',
         'Il downloader preserva il sorgente locale del modello già adattato.'],
        'get_json legge i metadati remoti; main seleziona soltanto le risorse necessarie. L’hash è un’impronta dei byte, non una garanzia sulla qualità dei dati. I file completati vengono rinominati soltanto dopo la verifica.',
        estratto('strumenti/download_resources.py', 'main', 'if partial.stat()', 'partial.replace(target)'), 'strumenti/download_resources.py · main()')
    add('5 · Lettura di un oggetto',
        ['leggi_labels legge soltanto le etichette, utili per filtrare le categorie.',
         'leggi_campione controlla split e indice.',
         'Legge il row group necessario e seleziona una sola riga.',
         'Restituisce punti float32 ed etichetta numerica.'],
        'Parquet organizza le righe in gruppi. Non carichiamo l’intero dataset in RAM per ogni selezione, ma il gruppo contenente il campione. L’etichetta è il riferimento con cui confrontare la predizione.',
        estratto('inferenza.py', 'leggi_campione', 'for group', "return np.asarray"), 'inferenza.py · leggi_campione()')
    add('6 · Preprocessing: rendere confrontabili le forme',
        ['Da 2.048 a 1.024 punti, con seed 42.', 'Centratura: sottrazione della media XYZ.',
         'Scala uniforme: divisione per il raggio massimo.', 'Output: matrice (1.024, 3), centrata nella sfera unitaria.'],
        'La normalizzazione riduce differenze di posizione e dimensione mantenendo le proporzioni. Se ci sono meno di 1.024 punti si campiona con ripetizione; una nuvola di raggio zero non viene divisa per zero. Il seed rende ripetibile il sottoinsieme scelto.',
        estratto('inferenza.py', 'prepara_punti', 'rng =', 'return selected'), 'inferenza.py · prepara_punti()')
    add('7 · Caricare i pesi senza addestrare',
        ['Ricreiamo PointNetClassification con 40 uscite.', 'Carichiamo i pesi su CPU con weights_only=True.',
         'strict=True richiede corrispondenza completa tra architettura e checkpoint.',
         'eval() disattiva dropout e aggiornamenti delle statistiche BatchNorm.'],
        'eval non congela da solo i gradienti: per l’inferenza usiamo anche inference_mode. La CPU permette di usare il progetto senza GPU. I pesi non vengono modificati durante l’esplorazione.',
        estratto('inferenza.py', 'carica_modello', 'torch.set_num_threads', 'return model.eval'), 'inferenza.py · carica_modello()')
    add('8 · Il viaggio di un oggetto nella rete',
        ['Input: (B, 3, N), con B=1 e N=1.024 nell’app.',
         'T-Net 3×3 → feature 64 → T-Net 64×64.',
         'Feature per punto: 64 → 128 → 1.024.',
         'Max pooling → descrittore globale di 1.024 valori.',
         'Classificatore: 1.024 → 512 → 256 → 40 logits.'],
        'B è il numero di oggetti nel batch. La dimensione N identifica i punti. Le dimensioni delle feature sono canali numerici appresi, non punti aggiunti né categorie intermedie.')
    add('9 · Prima trasformazione: T-Net sulle coordinate',
        ['Una piccola rete predice una matrice 3×3 per ciascun oggetto.',
         'torch.bmm applica la matrice agli XYZ del batch.',
         'I punti restano gli stessi: cambia la rappresentazione delle coordinate.',
         'La matrice può cambiare scala e forma; non è necessariamente una rotazione.'],
        'TNet usa convoluzioni condivise, max pooling e livelli lineari per produrre la matrice. All’inizializzazione la sua uscita è l’identità. Nel nostro checkpoint la trasformazione è già appresa: per ogni nuovo input viene calcolata una matrice senza effettuare training.',
        estratto('modello/pointnet_modelnet40.py', 'forward', 'bs =', 'return x.view'), 'modello/pointnet_modelnet40.py · TNet.forward()')
    add('10 · Feature condivise e seconda T-Net',
        ['Conv1d con kernel 1 applica gli stessi pesi a ciascun punto.',
         'BatchNorm e ReLU accompagnano l’estrazione delle feature.',
         'La T-Net 64×64 opera nello spazio delle feature.',
         'I canali successivi arrivano a 1.024 valori per punto.'],
        'Non si usa una griglia di voxel. La stessa funzione viene applicata a ogni punto; poi il pooling aggrega l’insieme. La trasformazione 64×64 non è visualizzabile direttamente come una trasformazione XYZ. Le feature non hanno automaticamente il significato di ala, gamba o ruota.',
        'self.conv1 = nn.Conv1d(3, 64, 1)\nself.feature_transform = TNet(k=64)\nself.conv3 = nn.Conv1d(64, 64, 1)\nself.conv4 = nn.Conv1d(64, 128, 1)\nself.conv5 = nn.Conv1d(128, 1024, 1)', 'modello/pointnet_modelnet40.py · PointNetClassification.__init__() (estratti)')
    add('11 · Max pooling: da punti a oggetto',
        ['Prima: tensore (B, 1.024 canali, N punti).',
         'Per ogni canale scegliamo il massimo fra tutti i punti.',
         'Dopo: un vettore (B, 1.024) che descrive l’oggetto.',
         'Riordinare gli stessi punti non cambia i massimi.'],
        'Esempio: tre punti con feature [0,2; 0,9], [0,8; 0,3], [0,5; 0,6] producono [0,8; 0,9]. I due massimi possono provenire da punti diversi. L’invarianza riguarda la permutazione degli stessi punti, non ogni possibile nuovo campionamento casuale.',
        'x = torch.max(x, dim=2, keepdim=False)[0]', 'modello/pointnet_modelnet40.py · PointNetClassification.forward()')
    add('12 · Dal descrittore alla categoria',
        ['fc1 e fc2 riducono il descrittore a 512 e poi 256 valori.',
         'fc3 produce 40 logits: punteggi grezzi delle categorie.',
         'Softmax trasforma i logits in punteggi relativi che sommano a 1.',
         'argmax sceglie la categoria con il punteggio più alto.'],
        'Una softmax alta non garantisce una classificazione corretta. La rete sceglie comunque tra le 40 classi anche quando l’oggetto appartiene a una categoria sconosciuta. Il dropout è presente nel codice, ma è disattivato in eval.',
        'x = F.relu(self.bn6(self.fc1(x)))\nx = F.relu(self.bn7(self.fc2(x)))\nx = F.dropout(x, p=self.dropout, training=self.training)\nx = self.fc3(x)', 'modello/pointnet_modelnet40.py · PointNetClassification.forward() (estratto)')
    add('13 · analizza(): osservare senza cambiare il forward',
        ['Trasponiamo gli XYZ e aggiungiamo la dimensione batch.',
         'Un forward hook su bn5 cattura le feature prima del pooling.',
         'Applichiamo ReLU anche ai valori osservati, come nel forward.',
         'finally rimuove sempre l’hook.'],
        'La rete restituisce logits e matrici di trasformazione. L’hook permette di leggere un livello interno senza riscrivere l’architettura; detach e clone separano il dato osservato dal calcolo. La rimozione evita di accumulare osservatori tra analisi successive.',
        estratto('inferenza.py', 'analizza', 'def osserva_feature', 'hook.remove()'), 'inferenza.py · analizza()')
    add('14 · Cosa mostrano i contributi al max pooling',
        ['Calcoliamo il massimo di ogni canale.',
         'argmax individua il punto vincente nei canali attivi.',
         'bincount conta quante volte ciascun punto fornisce il massimo.',
         'Il colore non rappresenta segmentazione o importanza causale.'],
        'In caso di parità viene contato il primo punto. I canali con massimo zero vengono esclusi. Il descrittore globale ha 1.024 valori; una sua eventuale griglia 32×32 sarebbe soltanto una disposizione grafica dei numeri, non un’immagine della forma.',
        estratto('inferenza.py', 'analizza', "features = activations", 'counts ='), 'inferenza.py · analizza()')
    add('15 · L’interfaccia: scegliere, osservare, classificare',
        ['Streamlit gestisce categoria, split e cambio di esempio.',
         'Plotly mostra la nuvola ruotabile e le cinque categorie principali.',
         'Verde/rosso confronta la predizione del singolo esempio con la sua etichetta.',
         'L’approfondimento mostra input, trasformazione e contributi al pooling.'],
        'Non stiamo misurando l’accuracy di un’intera categoria con il pallino: valutiamo solo l’esempio corrente. figura mantiene le proporzioni con aspectmode=data. Le top 5 non vengono rinormalizzate: possono sommare a meno del 100%.',
        estratto('app.py', 'figura'), 'app.py · figura()')
    add('16 · Cache e stato dell’app',
        ['cache_data conserva etichette, campioni e risultati per split/indice.',
         'Il seed fisso rende coerenti le analisi ripetute.',
         'session_state ricorda l’esempio corrente e la selezione analizzata.',
         'Cambiare oggetto impedisce di mostrare il risultato precedente.'],
        'Streamlit riesegue lo script quando si interagisce con i controlli. La cache riduce il lavoro ripetuto. Il modello viene caricato quando la funzione di risultato entra effettivamente in esecuzione; qui non è usata una cache_resource del modello.',
        estratto('app.py', 'risultato_cached'), 'app.py · risultato_cached() (decorata con @st.cache_data)')
    add('17 · Verifiche del progetto base',
        ['Forma (1.024, 3), centro vicino a zero e raggio unitario.',
         'Ripetibilità del preprocessing e rifiuto di input non validi.',
         'Descrittore visualizzato uguale all’ingresso reale di fc1.',
         'Softmax, punti trasformati e conteggio dei massimi coerenti.',
         'Predizione stabile permutando gli stessi punti.'],
        'strumenti/test_inferenza.py usa il modello reale. Il pre-hook su fc1 cattura il descrittore effettivamente usato dal classificatore: questa verifica controlla che la spiegazione visiva corrisponda al calcolo, non soltanto che abbia la forma giusta.',
        'np.testing.assert_allclose(result[\'global_features\'], captured[\'descriptor\'][0])\nwith torch.inference_mode():\n    permuted, _, _ = self.model(tensor.flip(dims=[2]))\ntorch.testing.assert_close(logits, permuted, atol=1e-4, rtol=1e-4)', 'strumenti/test_inferenza.py · test_inspection_matches_real_forward() (estratti)')
    add('18 · Il training completo presente nel sorgente originale',
        ['ModelNet40Dataset campiona, centra e normalizza gli oggetti.',
         'augment_pointcloud aggiunge rotazioni attorno a Z e rumore.',
         'train_epoch aggiorna tutta la rete; evaluate misura senza gradienti.',
         'Queste funzioni sono fornite con il modello, ma non sono chiamate dall’app.'],
        'La loss originale combina classificazione e regolarizzazione delle due T-Net. La penalità favorisce l’ortogonalità senza imporla esattamente. Il main originale include scheduler, checkpoint e upload opzionale. Non attribuiamo al progetto locale un riaddestramento completo non documentato.',
        estratto('modello/pointnet_modelnet40.py', 'train_epoch', 'cls_loss =', 'optimizer.step()'), 'modello/pointnet_modelnet40.py · train_epoch()')
    add('19 · Dimostrazione pratica della parte base',
        ['Avviare Streamlit dalla radice del progetto.',
         'Selezionare una categoria e ruotare l’oggetto originale.',
         'Premere “Invia alla rete” e confrontare predizione ed etichetta.',
         'Aprire l’approfondimento e seguire input → T-Net → max pooling.'],
        'Usare almeno un esempio corretto e uno errato per mostrare che il punteggio softmax non è una garanzia. Il test numerico si avvia con discovery dalla cartella strumenti, coerentemente con la posizione attuale dei test.',
        '.\\.venv\\Scripts\\python.exe -m streamlit run app.py\n.\\.venv\\Scripts\\python.exe -m unittest discover -s strumenti -p test_inferenza.py -v', 'Comandi PowerShell · dalla radice')
    add('20 · Estensione finale: perché i cavalli?',
        ['horse è assente dalle 40 categorie originali.',
         'Il modello iniziale può assegnare un cavallo solo a una vecchia classe.',
         'Domanda: quanto aiuta aumentare i cavalli annotati?',
         'Esperimento separato: 41 classi, modificando solo l’ultimo livello.'],
        'La nuova classe ha indice 40 perché gli indici precedenti sono 0–39. Questa è una forma limitata di fine-tuning supervisionato, spesso descritta come addestramento di una testa lineare su feature congelate. L’app principale conserva le sue 40 uscite.')
    add('21 · Acquisire i cavalli da Objaverse',
        ['Le annotazioni LVIS individuano gli ID della categoria horse.',
         'Scarichiamo solo le mesh selezionate, con limite di 12 MB per GLB.',
         'trimesh campiona 2.048 punti sulla superficie, pesando le aree.',
         'Salviamo XYZ .npy e manifest con origine e hash.'],
        'scarica_cavalli.py scarta geometrie senza triangoli o coordinate non valide. Il seed deriva dall’ID del modello. Il download usa quattro thread, ma il manifest è aggiornato dal thread principale. Le etichette LVIS non sono state revisionate manualmente.',
        estratto('esperimento_horse/scarica_cavalli.py', 'convert', 'scene =', 'np.save('), 'esperimento_horse/scarica_cavalli.py · convert()')
    add('22 · Split e confronto equo',
        [f'{len(protocol["horse_train_ids"])} cavalli nel pool di training; {len(protocol["horse_test_ids"])} cavalli nel test fisso.',
         'Deduplicazione per hash del GLB e split deterministico.',
         'Replay: 5 oggetti per vecchia classe, 200 totali.',
         'Test: 400 oggetti ModelNet40 + 20 cavalli, sempre identici.',
         'Budget eseguiti: 1, 5, 10, 25, 50; 100 saltato per dati insufficienti.'],
        'La deduplicazione riconosce file identici; non elimina necessariamente mesh geometricamente simili salvate in modi diversi. Il replay riporta nel training esempi delle vecchie categorie per limitare la perdita di capacità. Il numero sull’asse X conta solo i cavalli, non i 200 esempi di replay.',
        estratto('esperimento_horse/esperimento.py', 'split_horses', 'unique =', 'return [rows'), 'esperimento_horse/esperimento.py · split_horses()')
    add('23 · Congelare la rete e riusare le feature',
        ['carica_modello() imposta eval; requires_grad_(False) congela i parametri.',
         'Un pre-hook su fc3 legge il vettore di 256 valori.',
         'Preprocessing identico all’app, con seed 42.',
         'Le feature vengono calcolate una volta, a batch di 8 oggetti.'],
        'Il descrittore dopo pooling ha 1.024 valori; l’embedding usato dall’esperimento ha 256 valori perché è dopo fc1 e fc2. extract osserva le feature; il congelamento dei parametri avviene nel main prima della chiamata. BatchNorm non aggiorna le statistiche e dropout è disattivato.',
        estratto('esperimento_horse/esperimento.py', 'extract', 'captured =', 'return torch.cat'), 'esperimento_horse/esperimento.py · extract()')
    add('24 · Da 40 a 41 uscite',
        ['Creiamo nn.Linear(256, 41).', 'Copiamo pesi e bias delle prime 40 classi.',
         'La nuova riga horse viene inizializzata con seed controllato.',
         'Durante il fine-tuning si aggiornano tutte le 41 righe della nuova testa.'],
        'I livelli precedenti restano congelati, ma le 40 righe copiate non restano fisse: anche loro vengono ottimizzate. Per questo le prestazioni sulle vecchie classi possono cambiare. La testa originale rimane separata e non viene modificata.',
        estratto('esperimento_horse/esperimento.py', 'fit_head', 'torch.manual_seed', 'optimizer ='), 'esperimento_horse/esperimento.py · fit_head()')
    add('25 · Come addestriamo l’ultimo livello',
        ['Adam con learning rate 0,001; 300 passi per ogni prova.',
         'Loss = 50% cavalli + 50% replay delle vecchie classi.',
         'Tre seed: 11, 22, 33; supporti annidati per ciascun seed.',
         'Ogni budget riparte dall’originale, non dalla prova precedente.'],
        'Ogni passo usa tutte le feature del supporto e del replay. Qui 300 indica passi di ottimizzazione, non epoche di un training completo di PointNet. I sottoinsiemi annidati includono il supporto piccolo in quello più grande. Non si scelgono iperparametri in base al test.',
        estratto('esperimento_horse/esperimento.py', 'fit_head', 'for _ in range', 'return head.eval'), 'esperimento_horse/esperimento.py · fit_head()')
    add('26 · Quali metriche controllare',
        ['Recall horse: quota di cavalli del test riconosciuti.',
         'Precision horse: quota di veri cavalli tra le predizioni horse.',
         'Accuracy vecchie classi: conservazione delle capacità originali.',
         'Accuracy totale, macro accuracy e old false horse rate completano la lettura.'],
        'Un modello che risponde sempre cavallo ha recall 100% ma precision bassa e accuracy delle vecchie classi zero. Il test contiene 400 vecchi oggetti contro 20 cavalli, quindi l’accuracy complessiva pesa soprattutto le vecchie categorie. La macro accuracy dà lo stesso peso a ogni classe.',
        estratto('esperimento_horse/esperimento.py', 'metrics'), 'esperimento_horse/esperimento.py · metrics()')
    # La tabella usa le medie delle misure reali, incluse precision e falsi positivi.
    fields = ['horse_recall', 'horse_precision', 'old_accuracy', 'accuracy', 'old_false_horse_rate']
    table = ['| Cavalli | Recall horse | Precision horse | Accuracy vecchie | Accuracy totale | Vecchi → horse |',
             '|---:|---:|---:|---:|---:|---:|']
    table_html = '<table><thead><tr>' + ''.join(f'<th>{x}</th>' for x in ['Cavalli', 'Recall horse', 'Precision horse', 'Acc. vecchie', 'Acc. totale', 'Vecchi → horse']) + '</tr></thead><tbody>'
    averages = []
    for n in protocol['available']:
        group = [r for r in measurements if int(r['shots']) == n]
        values = [mean(float(r[f]) for r in group)*100 for f in fields]
        averages.append((n, values))
        cells = [str(n)] + [f'{v:.2f}%'.replace('.', ',') for v in values]
        table.append('| ' + ' | '.join(cells) + ' |')
        table_html += '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
    table_html += '</tbody></table>'
    add('27 · Risultati realmente misurati',
        ['Medie dei tre seed; esecuzione 20261001_181025.',
         'Baseline: accuracy vecchie 78,00%; accuracy totale 74,29%; recall horse 0%.',
         'Più cavalli migliorano la recall, ma aumentano anche i falsi positivi.'],
        'Con 50 esempi la recall raggiunge 95%, ma la precision media è circa 33,76%: molte predizioni cavallo riguardano vecchi oggetti. L’accuracy vecchie scende a circa 71,33%, rispetto al 78% della baseline. Non possiamo quindi descrivere l’esperimento come un miglioramento uniforme. La baseline è misurata sullo stesso sottoinsieme, non sull’intero ModelNet40.', extra=table_html)
    add('28 · Curva del compromesso',
        ['La recall cresce dal 23,33% al 95,00% in questa esecuzione.',
         'L’accuracy delle vecchie classi passa dal 74,17% al 71,33% tra i budget.',
         'Le barre nel grafico originale rappresentano deviazioni standard tra seed.',
         'Con 20 cavalli di test ogni singola recall cambia a passi di 5 punti percentuali.'],
        'L’andamento crescente è un risultato di queste prove, non una proprietà garantita. Le deviazioni standard non sono intervalli di confidenza. Il grafico offline completo è salvato nella cartella dei risultati.',
        extra='<p><a href="../esperimento_horse/risultati/20261001_181025/curva_accuracy.html">Apri il grafico interattivo dei risultati</a></p>')
    add('29 · Riproducibilità, dashboard e limiti',
        ['protocollo.json e supporti.json registrano split, impostazioni e ID.',
         'misure.csv, predizioni_*.npz e head_*.pt conservano risultati e teste.',
         'visualizza.py mostra misure e predizioni già salvate; non riaddestra.',
         'Dominio Objaverse diverso da ModelNet40; etichette non revisionate.',
         'I risultati non dimostrano generalizzazione ai cavalli reali.'],
        'Il controllo SHA256 finale verifica che il checkpoint base sia rimasto invariato. I checkpoint head contengono solo l’ultimo livello, da abbinare al modello originale. I test horse verificano split disgiunti, ripetibilità, conservazione della testa originale e metriche che penalizzano la risposta costante horse. La dashboard usa l’ultima cartella con misure e il primo seed per il singolo oggetto.',
        '.\\.venv\\Scripts\\python.exe -m streamlit run esperimento_horse/visualizza.py --server.port 8502\n.\\.venv\\Scripts\\python.exe -m unittest discover -s esperimento_horse -p test_esperimento.py -v', 'Comandi PowerShell · dashboard e test dell’estensione')

    # HTML autonomo: nessuna dipendenza da CDN, navigazione da tastiera e stampa PDF.
    sections = []
    markdown = ['# Progetto 1 — Presentazione e note per l’esposizione',
                'Aprire [Presentazione_Progetto_1.html](Presentazione_Progetto_1.html) nel browser. Frecce per cambiare slide; N per le note; F per lo schermo intero. La stampa include tutte le slide.',
                'I frammenti provengono dal codice locale; dove indicato sono estratti selezionati.']
    for i, s in enumerate(slides, 1):
        body = '<ul>' + ''.join(f'<li>{html.escape(b)}</li>' for b in s['bullets']) + '</ul>'
        if s['code']:
            body += f'<p class="source">{html.escape(s["source"])}</p><pre><code>{html.escape(s["code"])}</code></pre>'
        body += s['extra']
        sections.append(f'<section class="slide"><span class="eyebrow">3D Perception · Progetto 1</span><h1>{html.escape(s["title"])}</h1>{body}<aside>{html.escape(s["notes"])}</aside><footer>{i:02d} / {len(slides):02d}</footer></section>')
        markdown += [f'\n## Slide {i} — {s["title"]}', '\n' + '\n'.join('- ' + b for b in s['bullets'])]
        if s['code']:
            markdown += ['\n**Codice:** ' + s['source'], '\n```python\n' + s['code'] + '\n```']
        if s['title'] == '27 · Risultati realmente misurati':
            markdown.append('\n' + '\n'.join(table))
        markdown += ['\n**Note per l’esposizione:** ' + s['notes']]
    css = '''*{box-sizing:border-box}body{margin:0;background:#0b1425;color:#edf4ff;font-family:Segoe UI,Arial,sans-serif}.slide{display:none;min-height:100vh;padding:40px 6vw 70px;position:relative}.slide.active{display:block}.eyebrow{color:#62d6c5;text-transform:uppercase;letter-spacing:3px;font-size:13px}h1{font-size:clamp(26px,3vw,42px);margin:15px 0 24px;max-width:1150px}ul{padding-left:26px;max-width:1200px}li{font-size:clamp(17px,1.7vw,24px);line-height:1.45;margin:9px 0}pre{background:#14243b;border-left:4px solid #62d6c5;padding:16px 20px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:clamp(12px,1.15vw,17px);line-height:1.4;border-radius:5px;max-width:1280px}.source{color:#8faccd;font-size:14px;margin:20px 0 8px}aside{display:none;background:#20334b;padding:16px;line-height:1.6;margin-top:18px}.notes aside{display:block}footer{position:absolute;bottom:24px;right:6vw;color:#8faccd}nav{position:fixed;bottom:16px;left:6vw;display:flex;gap:8px;z-index:2}button,select{background:#20334b;color:#fff;border:1px solid #466484;padding:8px 12px;border-radius:5px;cursor:pointer}a{color:#62d6c5}table{border-collapse:collapse;width:100%;margin-top:28px;font-size:20px}td,th{padding:14px;border-bottom:1px solid #466484;text-align:right}th{color:#62d6c5;font-size:16px}@media print{body{background:white;color:#17243b}nav{display:none}.slide,.slide.active{display:block;min-height:0;height:auto;page-break-after:always;padding:12mm}h1{font-size:26px}li{font-size:16px}pre{font-size:11px;background:#eef3f8;color:#17243b}aside{display:none!important}footer{position:static;text-align:right}td,th{font-size:13px;padding:8px}a{color:#17243b}}@page{size:A4 landscape;margin:8mm}'''
    js = '''const slides=[...document.querySelectorAll('.slide')],select=document.querySelector('select');let current=0;slides.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+' · '+s.querySelector('h1').textContent;select.append(o)});function show(n){current=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('active',i===current));select.value=current;location.hash=current+1;window.scrollTo(0,0)}function notes(){document.body.classList.toggle('notes')}function full(){if(!document.fullscreenElement)document.documentElement.requestFullscreen();else document.exitFullscreen()}document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(current+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(current-1)}if(e.key.toLowerCase()==='n')notes();if(e.key.toLowerCase()==='f')full()});show((parseInt(location.hash.slice(1))||1)-1);'''
    page = '<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Progetto 1 · PointNet</title><style>' + css + '</style></head><body>' + ''.join(sections) + '<nav><button onclick="show(current-1)" aria-label="Slide precedente">←</button><button onclick="show(current+1)" aria-label="Slide successiva">→</button><select onchange="show(Number(this.value))" aria-label="Seleziona slide"></select><button onclick="notes()">Note</button><button onclick="full()">Schermo intero</button><button onclick="window.print()">PDF / Stampa</button></nav><script>' + js + '</script></body></html>'
    (OUT / 'Presentazione_Progetto_1.html').write_text(page, encoding='utf-8')
    (OUT / 'Presentazione_Progetto_1.md').write_text('\n'.join(markdown) + '\n', encoding='utf-8')
    print(f'Generate {len(slides)} slide in {OUT}')


if __name__ == '__main__':
    main()
