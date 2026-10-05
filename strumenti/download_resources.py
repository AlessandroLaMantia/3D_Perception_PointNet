"""Scarica modello e point cloud da Hugging Face, verificando dimensioni e SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request

# Risorse nella radice del progetto, un livello sopra strumenti/.
ROOT = Path(__file__).resolve().parents[1]


# Legge metadati remoti per conoscere revisione, dimensioni e hash prima del download.
# Scopo: Legge i metadati Hugging Face necessari per fissare revisione e file da acquisire.
def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


# Scarica solo le risorse selezionate, verifica i file e conserva la provenienza.
# Scopo: Scarica le risorse selezionate e verifica dimensioni e hash prima di registrarne l’origine.
def main():
    # Il manifest conserva la provenienza dei download, non dei successivi edit.
    manifest = []
    manifest_path = ROOT / 'provenienza.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    sources = [
        ('model', 'DavidHanSZ/pointnet-modelnet40', 'modello', {
            'README.md': 'README_originale.md', 'config.json': 'config.json',
            'pointnet_modelnet40.py': 'pointnet_modelnet40.py',
            'pytorch_model.bin': 'pytorch_model.bin'}),
        ('dataset', 'jxie/modelnet40-2048', 'dati', None),
    ]
    for kind, repo, folder, names in sources:
        api_kind = 'models' if kind == 'model' else 'datasets'
        info = get_json(f'https://huggingface.co/api/{api_kind}/{repo}')
        # Blocca ogni download a una revisione precisa del repository remoto.
        revision = info['sha']
        entries = get_json(f'https://huggingface.co/api/{api_kind}/{repo}/tree/{revision}?recursive=true')
        selected = [e for e in entries if e['type'] == 'file' and
                    (e['path'] in names if names else e['path'].endswith('.parquet') or e['path'] == 'README.md')]
        if kind == 'dataset' and sum(e['size'] for e in selected) > 400_000_000:
            raise RuntimeError('Il dataset supera 400 MB')
        for entry in selected:
            remote = entry['path']
            local = names[remote] if names else ('README_originale.md' if remote == 'README.md' else ('test.parquet' if '/test-' in remote else 'train.parquet'))
            target = ROOT / folder / local
            target.parent.mkdir(parents=True, exist_ok=True)
            prefix = 'datasets/' if kind == 'dataset' else ''
            url = f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{remote}'
            # L'hash pubblicato da Hugging Face verifica pesi e dataset binari.
            expected_hash = entry.get('lfs', {}).get('oid')
            # Questo sorgente e ora commentato e adattato per l'inferenza locale.
            # Conserva sia le modifiche sia il record del suo download originale.
            if remote == 'pointnet_modelnet40.py' and target.exists():
                record = next((item for item in previous if item['file'] == f'{folder}/{local}'), None)
                if record is not None:
                    manifest.append(record)
                    print(f'Preservato sorgente locale {folder}/{local}', flush=True)
                    continue
            if not target.exists():
                print(f'Scarico {folder}/{local}: {entry["size"]/1e6:.2f} MB', flush=True)
                partial = target.with_suffix(target.suffix + '.part')
                with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > entry['size']:
                            raise RuntimeError('Dimensione superiore a quella dichiarata')
                        output.write(chunk)
                if partial.stat().st_size != entry['size']:
                    raise RuntimeError('Download incompleto')
                with partial.open('rb') as handle:
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                if expected_hash and digest != expected_hash:
                    raise RuntimeError('SHA256 non corrispondente')
                partial.replace(target)
            with target.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if target.stat().st_size != entry['size'] or (expected_hash and digest != expected_hash):
                raise RuntimeError(f'File non valido: {target}')
            manifest.append(dict(file=f'{folder}/{local}', bytes=target.stat().st_size,
                                 sha256=digest, source=url, revision=revision))
            print(f'Verificato {folder}/{local}', flush=True)
    (ROOT / 'provenienza.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Download e verifica completati.', flush=True)


if __name__ == '__main__':
    main()
