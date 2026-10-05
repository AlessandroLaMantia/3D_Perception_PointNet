"""Scarica cavalli Objaverse-LVIS e campiona superfici, senza alterare i dati originali."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
BASE = 'https://huggingface.co/datasets/allenai/objaverse/resolve/main/'


# Limita i byte sia tramite header sia durante la lettura; evita download troppo grandi.
# Scopo: Controlla il limite di byte prima e durante il download del singolo file.
def fetch(url, limit):
    with urllib.request.urlopen(url, timeout=90) as response:
        if int(response.headers.get('Content-Length', '0')) > limit:
            raise ValueError('File superiore al limite per singolo modello')
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('File superiore al limite per singolo modello')
    return data


# Scopo: Seleziona horse dalle annotazioni LVIS e registra le point cloud scaricate nel manifest.
def main():
    """Scarica solo i cavalli LVIS e converte le mesh in point cloud XYZ.

    L'obiettivo e' acquisire un mini-dataset per lo studio del fine-tuning few-shot:
    manteniamo solo le nuvole campionate e scartiamo i modelli 3D pesanti. In questo
    modo il progetto resta leggero e facilmente riproducibile in un workspace locale.
    """
    folder = HERE / 'dati'
    folder.mkdir(parents=True, exist_ok=True)
    # Metadati piccoli; gli archivi completi di Objaverse non vengono scaricati.
    # Ci servono solo i file di annotazione per trovare i modelli "horse" e i path dei file .glb.
    annotations = json.loads(gzip.decompress(fetch(BASE + 'lvis-annotations.json.gz', 10_000_000)))
    paths = json.loads(gzip.decompress(fetch(BASE + 'object-paths.json.gz', 20_000_000)))
    ids = sorted(annotations['horse'])
    rng = np.random.default_rng(2026)
    rng.shuffle(ids)
    manifest_path = folder / 'manifest.json'
    records = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    # In un clone senza dataset il manifest può esserci, ma gli XYZ vanno riscaricati.
    # Non consideriamo disponibili file mancanti o con un hash diverso dal registrato.
    records = [r for r in records
               if (folder / r['file']).is_file()
               and hashlib.sha256((folder / r['file']).read_bytes()).hexdigest() == r['points_sha256']]
    existing = {r['id'] for r in records}
    # Trasforma una mesh triangolare in 2048 punti campionati sulla superficie.
    # Scopo: Converte una scena GLB in triangoli, campiona la superficie e salva solo gli XYZ.
    def convert(uid):
        try:
            path = paths[uid]
            url = BASE + path
            # I GLB vengono elaborati in RAM; su disco restano solo ~25 KB di XYZ.
            blob = fetch(url, 12_000_000)
            scene = trimesh.load(io.BytesIO(blob), file_type='glb', force='scene', process=False)
            mesh = scene.to_geometry()
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ValueError('Nessuna superficie triangolare')
            points, _ = trimesh.sample.sample_surface(mesh, 2048, seed=int(uid[:8], 16))
            points = np.asarray(points, dtype=np.float32)
            if not np.isfinite(points).all() or np.ptp(points, axis=0).max() == 0:
                raise ValueError('Geometria non valida')
            target = folder / f'{uid}.npy'
            np.save(target, points, allow_pickle=False)
            return dict(id=uid, file=target.name, source=url,
                                mesh_sha256=hashlib.sha256(blob).hexdigest(),
                                points_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                label_source=BASE + 'lvis-annotations.json.gz')
        except Exception as exc:
            print(f'Saltato {uid}: {type(exc).__name__}: {exc}', flush=True)
            return None
    # Solo il thread principale aggiorna il manifest, evitando scritture concorrenti.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(convert, uid) for uid in ids if uid not in existing]
        for future in as_completed(futures):
            record = future.result()
            if record is not None:
                records.append(record)
                manifest_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
                print(f'{len(records)}/{len(ids)}: {record["id"]}', flush=True)
    print(f'Cavalli disponibili: {len(records)}. Etichette LVIS, non revisionate manualmente.', flush=True)


if __name__ == '__main__':
    main()
