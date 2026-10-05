"""Lettura locale, preprocessing e ispezione delle attivazioni di PointNet.

Questo modulo incapsula il flusso base del progetto: carica il modello preaddestrato,
applica il preprocessing dei point cloud e permette di ispezionare le attivazioni
interne della rete. E' il punto di partenza per tutte le analisi successive, inclusi
il confronto train/test e il fine-tuning few-shot su horse.
"""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from modello.pointnet_modelnet40 import PointNetClassification

ROOT = Path(__file__).resolve().parent

# Ordine standard ModelNet40, indici 0..39; fonte riportata nel README.
# Questa tabella ci permette di convertire il codice numerico della classe in un nome
# leggibile in italiano e inglese per la presentazione e l'analisi dei risultati.
CATEGORIE = [
    ('airplane', 'aereo'), ('bathtub', 'vasca da bagno'), ('bed', 'letto'),
    ('bench', 'panchina'), ('bookshelf', 'libreria'), ('bottle', 'bottiglia'),
    ('bowl', 'ciotola'), ('car', 'automobile'), ('chair', 'sedia'), ('cone', 'cono'),
    ('cup', 'tazza'), ('curtain', 'tenda da finestra'), ('desk', 'scrivania'),
    ('door', 'porta'), ('dresser', 'cassettiera'), ('flower_pot', 'vaso da fiori'),
    ('glass_box', 'scatola di vetro'), ('guitar', 'chitarra'), ('keyboard', 'tastiera'),
    ('lamp', 'lampada'), ('laptop', 'portatile'), ('mantel', 'cornice del camino'),
    ('monitor', 'monitor'), ('night_stand', 'comodino'), ('person', 'persona'),
    ('piano', 'pianoforte'), ('plant', 'pianta'), ('radio', 'radio'),
    ('range_hood', 'cappa aspirante'), ('sink', 'lavandino'), ('sofa', 'divano'),
    ('stairs', 'scale'), ('stool', 'sgabello'), ('table', 'tavolo'),
    ('tent', 'tenda da campeggio'), ('toilet', 'WC'), ('tv_stand', 'mobile TV'),
    ('vase', 'vaso'), ('wardrobe', 'armadio'), ('xbox', 'Xbox'),
]


# Scopo: Associa l’indice del dataset al nome italiano e inglese della categoria.
def nome_classe(indice):
    """Converte un indice numerico delle classi in una stringa leggibile."""
    inglese, italiano = CATEGORIE[int(indice)]
    return f'{italiano} ({inglese})'


# Scopo: Legge la sola colonna label: il filtro delle classi non richiede le coordinate.
def leggi_labels(split):
    """Legge solo le etichette per filtrare gli oggetti senza caricare gli XYZ."""
    return pq.read_table(ROOT / 'dati' / f'{split}.parquet', columns=['label'])['label'].to_numpy()


# Scopo: Trova la riga nel Parquet e restituisce XYZ ed etichetta di riferimento.
def leggi_campione(split, indice):
    """Restituisce un punto cloud e la sua label senza caricare tutto il dataset in memoria."""
    if split not in ('train', 'test'):
        raise ValueError('Split non valido')
    file = pq.ParquetFile(ROOT / 'dati' / f'{split}.parquet')
    if not 0 <= indice < file.metadata.num_rows:
        raise IndexError('Indice fuori intervallo')
    # Salta i row group precedenti e decodifica solo quello del campione scelto.
    # In questo modo si evitano letture inutili e si mantiene il codice efficiente su CPU.
    for group in range(file.num_row_groups):
        count = file.metadata.row_group(group).num_rows
        if indice >= count:
            indice -= count
            continue
        row = file.read_row_group(group, columns=['inputs', 'label']).slice(indice, 1).to_pylist()[0]
        return np.asarray(row['inputs'], dtype=np.float32), int(row['label'])


# Il seed determina il sottoinsieme; centratura e scala uniforme conservano la forma.
# Scopo: Seleziona 1024 XYZ, sottrae il centro e normalizza la scala con seed ripetibile.
def prepara_punti(points, seed=42):
    """Replica il preprocessing del training; il seed rende stabile il campione.

    Il modello PointNet riceve coordinate XYZ centrate e normalizzate. La scelta casuale
    dei 1024 punti e la normalizzazione al raggio massimo rendono l'input stabile, ma
    comunque compatibile con il training originale della rete.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError('Servono punti XYZ finiti, forma (N, 3)')
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), 1024, replace=len(points) < 1024)
    # copy evita di alterare la nuvola originale durante centratura e riscalamento.
    selected = points[indices].copy()
    selected -= selected.mean(axis=0)
    radius = np.linalg.norm(selected, axis=1).max()
    if radius > 0:
        selected /= radius
    return selected


# Architettura e state_dict devono combaciare: strict=True rileva pesi incompatibili.
# Scopo: Ricrea l’architettura a 40 classi e carica il checkpoint per inferenza su CPU.
def carica_modello():
    """Carica il PointNet preaddestrato su ModelNet40 in modalità eval."""
    # CPU per funzionare anche senza GPU. eval disattiva dropout e aggiornamenti BN.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = PointNetClassification(num_classes=40)
    weights = torch.load(ROOT / 'modello' / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    return model.eval()


# Preprocessing, forward e raccolta delle attivazioni: restituisce dati numerici per le viste.
# Scopo: Coordina preprocessing e forward, poi prepara i dati delle visualizzazioni interne.
def analizza(model, points, seed=42):
    selected = prepara_punti(points, seed)
    # PyTorch vuole (batch, coordinate, punti), qui un solo oggetto.
    tensor = torch.from_numpy(selected.T.copy()).unsqueeze(0)
    activations = {}

    # Scopo: Osserva bn5 e replica la ReLU successiva per leggere le feature reali pre-pooling.
    def osserva_feature(module, inputs, output):
        # bn5 precede ReLU e max pooling nel forward originale.
        activations['features'] = torch.relu(output).detach().clone()

    # L'hook osserva un livello interno senza cambiare il risultato della rete.
    hook = model.bn5.register_forward_hook(osserva_feature)
    try:
        with torch.inference_mode():
            logits, transform, _ = model(tensor)
    finally:
        hook.remove()
    features = activations['features'][0].numpy()  # (1024 canali, 1024 punti)
    global_features = features.max(axis=1)
    # Per ogni canale attivo, identifica un punto che fornisce il massimo.
    # In caso di parita argmax sceglie il primo: non e una saliency causale.
    active = global_features > 0
    # Dopo ReLU un massimo nullo non segnala una feature attiva: non lo contiamo.
    winners = features[active].argmax(axis=1)
    counts = np.bincount(winners, minlength=len(selected))
    return {
        'input': selected,
        'aligned': (transform[0].numpy() @ selected.T).T,
        'probabilities': torch.softmax(logits, dim=1)[0].numpy(),
        'global_features': global_features,
        'max_counts': counts,
    }
