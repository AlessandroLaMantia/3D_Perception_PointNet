"""Few-shot class-incremental: fine-tuning del solo ultimo livello, da 40 a 41 classi."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
import plotly.graph_objects as go
from inferenza import carica_modello, prepara_punti


# Impronta del file per rilevare modifiche ai dati e ai pesi originali.
# Scopo: Calcola l’impronta SHA256 per controllare che dati e checkpoint siano invariati.
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


# Scopo: Deduplica file mesh identici e crea pool di training e test fisso disgiunti.
def split_horses(records, test_count=20):
    """Divide i cavalli in train/test dopo deduplicazione e split deterministico.

    L'obiettivo e' evitare contaminazione tra train e test: se due mesh hanno lo stesso
    hash, vengono contate come una sola identita. Questo fa aumentare la affidabilita del
    benchmark low-shot e riduce la probabilita di testare sullo stesso oggetto usato per
    il training.
    """
    # Identita e hash delle mesh: copie identiche non entrano in split diversi.
    unique = {}
    for row in records:
        unique.setdefault(row['mesh_sha256'], row)
    rows = sorted(unique.values(), key=lambda r: r['id'])
    if len(rows) <= test_count:
        raise ValueError(f'Servono piu di {test_count} cavalli distinti; trovati {len(rows)}')
    order = np.random.default_rng(2026).permutation(len(rows))
    return [rows[i] for i in order[test_count:]], [rows[i] for i in order[:test_count]]


# Campionamento bilanciato per classe: train per replay, test solo per valutazione.
# Scopo: Seleziona oggetti bilanciati per classe, separando replay train e valutazione test.
def old_examples(split, per_class):
    file = ROOT / 'dati' / f'{split}.parquet'
    labels = pq.read_table(file, columns=['label'])['label'].to_numpy()
    rng = np.random.default_rng(123 if split == 'train' else 456)
    ids = sorted(int(i) for c in range(40)
                 for i in rng.choice(np.flatnonzero(labels == c), per_class, replace=False))
    wanted = set(ids)
    points, targets, offset = [], [], 0
    for batch in pq.ParquetFile(file).iter_batches(batch_size=64):
        for local in range(batch.num_rows):
            if offset + local in wanted:
                row = batch.slice(local, 1).to_pylist()[0]
                points.append(np.asarray(row['inputs'], dtype=np.float32))
                targets.append(int(row['label']))
        offset += batch.num_rows
    return points, np.array(targets), ids


# Scopo: Legge le feature 256D prima di fc3; il main ha già congelato il modello in eval.
def extract(model, points):
    """Estraggo le feature 256-dim del backbone e congelo tutte le sue componenti.

    La rete originale termina con fc3, che produce 40 logits. Prima di quel layer,
    PointNet produce un embedding globale di dimensione 256. Estraendo questo vettore per
    ogni point cloud, possiamo addestrare solo l'ultimo layer per il nuovo oggetto.
    """
    captured = []
    # Scopo: Copia l’ingresso di fc3: è l’embedding dopo fc1/fc2, non il descrittore 1024D.
    def hook(module, args):
        captured.append(args[0].detach().cpu().clone())
    handle = model.fc3.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            for start in range(0, len(points), 8):
                x = np.stack([prepara_punti(p, 42).T for p in points[start:start+8]])
                model(torch.from_numpy(x))
    finally:
        handle.remove()
    return torch.cat(captured)


# Recall dei cavalli e precision vanno lette insieme alle prestazioni sulle vecchie classi.
# Scopo: Misura riconoscimento dei cavalli, falsi positivi e conservazione delle vecchie classi.
def metrics(pred, labels):
    pred, labels = np.asarray(pred), np.asarray(labels)
    horse = labels == 40
    tp = int(((pred == 40) & horse).sum())
    fp = int(((pred == 40) & ~horse).sum())
    return dict(accuracy=float((pred == labels).mean()),
                horse_recall=float((pred[horse] == 40).mean()),
                horse_precision=tp / (tp + fp) if tp + fp else 0.0,
                old_accuracy=float((pred[~horse] == labels[~horse]).mean()),
                old_false_horse_rate=float((pred[~horse] == 40).mean()),
                macro_accuracy=float(np.mean([(pred[labels == c] == c).mean() for c in range(41)])))


# Scopo: Ottimizza tutte le 41 righe della nuova testa; la testa originale resta separata.
def fit_head(original, support, replay, replay_y, seed, steps, lr):
    """Addestra solo il nuovo layer di classificazione per horse.

    Il backbone e' congelato. La nuova testa passa da 40 a 41 classi, mantenendo i pesi
    originali delle prime 40 righe e inizializzando la nuova classe horse con seed noto.
    """
    torch.manual_seed(seed)
    head = nn.Linear(256, 41)
    with torch.no_grad():
        head.weight[:40].copy_(original.weight)
        head.bias[:40].copy_(original.bias)
    # Ogni budget riparte dalla stessa inizializzazione, mai dalla prova precedente.
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        positives = torch.full((len(support),), 40, dtype=torch.long)
        # Stesso peso totale a cavalli e replay, anche con un solo cavallo.
        loss = .5 * nn.functional.cross_entropy(head(support), positives)
        loss += .5 * nn.functional.cross_entropy(head(replay), replay_y)
        loss.backward()
        optimizer.step()
    return head.eval()


# Orchestra split fisso, feature, budget indipendenti e salvataggio delle misure.
# Scopo: Orchestra test fisso, feature congelate e prove indipendenti per budget e seed.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots', nargs='+', type=int, default=[1, 5, 10, 25, 50, 100])
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--horse-test', type=int, default=20)
    parser.add_argument('--replay-per-class', type=int, default=5)
    parser.add_argument('--test-per-class', type=int, default=10)
    args = parser.parse_args()
    if min(args.shots + [args.steps, args.horse_test, args.replay_per_class, args.test_per_class]) < 1 or args.lr <= 0:
        parser.error('Quantita e learning rate devono essere positivi')
    manifest = HERE / 'dati' / 'manifest.json'
    if not manifest.exists():
        parser.error('Prima esegui esperimento_horse/scarica_cavalli.py')
    records = json.loads(manifest.read_text())
    for row in records:
        if sha(HERE / 'dati' / row['file']) != row['points_sha256']:
            raise ValueError(f'Point cloud modificata: {row["file"]}')
    train_h, test_h = split_horses(records, args.horse_test)
    available = [n for n in args.shots if n <= len(train_h)]
    skipped = [n for n in args.shots if n > len(train_h)]
    if not available:
        parser.error('Nessun budget compatibile con i cavalli disponibili')
    # Una cartella per esecuzione: nessun peso o risultato precedente sovrascritto.
    out = HERE / 'risultati' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    original_path = ROOT / 'modello' / 'pytorch_model.bin'
    original_hash = sha(original_path)
    model = carica_modello()
    # eval viene impostato dal loader; qui impediamo anche il calcolo dei gradienti
    # sui parametri originali. L'ottimizzatore riceverà solo la nuova testa.
    model.requires_grad_(False)
    print(f'Cavalli: {len(train_h)} training pool, {len(test_h)} test. Budget saltati: {skipped}', flush=True)
    old_train, old_train_y, train_ids = old_examples('train', args.replay_per_class)
    old_test, old_test_y, test_ids = old_examples('test', args.test_per_class)
    # Scopo: Carica gli XYZ locali dei cavalli con pickle disabilitato.
    def load_rows(rows):
        return [np.load(HERE / 'dati' / r['file'], allow_pickle=False) for r in rows]
    print('Estrazione delle feature congelate (CPU)...', flush=True)
    replay = extract(model, old_train)
    horse_train = extract(model, load_rows(train_h))
    evaluation = extract(model, old_test + load_rows(test_h))
    labels = np.concatenate([old_test_y, np.full(len(test_h), 40)])
    # La baseline usa gli stessi oggetti di test, ma le sue 40 uscite non includono horse.
    with torch.no_grad():
        baseline = metrics(model.fc3(evaluation).argmax(1).numpy(), labels)
    protocol = dict(config=vars(args), available=available, skipped=skipped,
                    original_weights_sha256=original_hash, horse_train_ids=[r['id'] for r in train_h],
                    horse_test_ids=[r['id'] for r in test_h], replay_train_ids=train_ids,
                    old_test_ids=test_ids, baseline_40_classes=baseline,
                    note='Solo ultimo Linear addestrato. Test fisso; nessun tuning sul test. Etichette LVIS non revisionate manualmente.')
    (out / 'protocollo.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (out / 'provenienza_dati.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    rows, supports = [], []
    for seed in args.seeds:
        order = np.random.default_rng(seed).permutation(len(train_h))
        for n in available:
            chosen = order[:n]  # sottoinsiemi annidati: 1 e incluso in 5, ecc.
            start = time.perf_counter()
            head = fit_head(model.fc3, horse_train[chosen], replay, torch.from_numpy(old_train_y).long(),
                            seed, args.steps, args.lr)
            with torch.no_grad():
                pred = head(evaluation).argmax(1).numpy()
            row = dict(shots=n, seed=seed, seconds=time.perf_counter()-start, **metrics(pred, labels))
            rows.append(row)
            supports.append(dict(shots=n, seed=seed, ids=[train_h[i]['id'] for i in chosen]))
            torch.save(dict(head_state_dict=head.state_dict(), original_weights_sha256=original_hash,
                            new_class='horse', new_class_id=40), out / f'head_{n}_{seed}.pt')
            np.savez_compressed(out / f'predizioni_{n}_{seed}.npz', pred=pred, labels=labels)
            print(f'{n} esempi, seed {seed}: accuracy {row["accuracy"]:.3f}, '
                  f'recall cavalli {row["horse_recall"]:.3f}, vecchie classi {row["old_accuracy"]:.3f}', flush=True)
    with (out / 'misure.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'supporti.json').write_text(json.dumps(supports, indent=2), encoding='utf-8')
    fig = go.Figure()
    summary = []
    for field, name in [('horse_recall', 'Cavalli riconosciuti (recall)'),
                        ('old_accuracy', 'Accuracy vecchie classi'), ('accuracy', 'Accuracy complessiva')]:
        means, stds = [], []
        for n in available:
            values = [r[field]*100 for r in rows if r['shots'] == n]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
            summary.append(dict(shots=n, metric=field, mean_percent=means[-1], std_percent=stds[-1]))
        fig.add_trace(go.Scatter(x=available, y=means, mode='lines+markers', name=name,
                                error_y=dict(type='data', array=stds)))
    fig.update_layout(title='Fine-tuning ultimo livello: 40 classi + horse',
                      xaxis=dict(title='Cavalli distinti nel training', type='category'),
                      yaxis=dict(title='Percentuale sul test fisso', range=[0, 100]))
    fig.add_annotation(text='Barre: deviazione standard tra seed. Non intervalli di confidenza. '
                       + (f'Non eseguiti: {skipped} (dati insufficienti).' if skipped else ''),
                       xref='paper', yref='paper', x=0, y=-.25, showarrow=False)
    fig.update_layout(margin=dict(b=120))
    fig.write_html(out / 'curva_accuracy.html', include_plotlyjs=True)
    (out / 'riepilogo.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    assert sha(original_path) == original_hash, 'I pesi originali sono cambiati!'
    print(f'Risultati: {out}', flush=True)


if __name__ == '__main__':
    main()
