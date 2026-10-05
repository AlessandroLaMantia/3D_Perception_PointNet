"""Aggiunge commenti didattici alle funzioni, verificando che l'AST non cambi.

Utility ripetibile: non duplica i commenti già inseriti e non modifica istruzioni.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMENTS = {
    'inferenza.py': {
        'nome_classe': 'Associa l’indice del dataset al nome italiano e inglese della categoria.',
        'leggi_labels': 'Legge la sola colonna label: il filtro delle classi non richiede le coordinate.',
        'leggi_campione': 'Trova la riga nel Parquet e restituisce XYZ ed etichetta di riferimento.',
        'prepara_punti': 'Seleziona 1024 XYZ, sottrae il centro e normalizza la scala con seed ripetibile.',
        'carica_modello': 'Ricrea l’architettura a 40 classi e carica il checkpoint per inferenza su CPU.',
        'analizza': 'Coordina preprocessing e forward, poi prepara i dati delle visualizzazioni interne.',
        'osserva_feature': 'Osserva bn5 e replica la ReLU successiva per leggere le feature reali pre-pooling.',
    },
    'app.py': {
        'labels_cached': 'Memorizza le etichette per evitare letture durante ogni interazione con Streamlit.',
        'campione_cached': 'Memorizza geometria ed etichetta usando split e indice come chiave.',
        'risultato_cached': 'Memorizza l’analisi del singolo oggetto: l’app non aggiorna i pesi.',
        'figura': 'Crea uno scatter XYZ ruotabile mantenendo le proporzioni della geometria.',
        'percentuale': 'Formatta il punteggio senza arrotondare a zero le alternative molto piccole.',
        'grafico_categorie': 'Mostra le cinque softmax maggiori sulla scala comune 0–100 senza rinormalizzarle.',
        'cambia_esempio': 'Avanza ciclicamente fra gli oggetti della classe prima del nuovo rerun.',
    },
    'modello/pointnet_modelnet40.py': {
        'TNet.__init__': 'Costruisce una mini-PointNet che inizialmente restituisce la matrice identità.',
        'TNet.forward': 'Riassume i punti e produce una matrice k×k distinta per ogni oggetto del batch.',
        'PointNetClassification.__init__': 'Definisce trasformazioni, MLP condivisi e classificatore per l’intero oggetto.',
        'PointNetClassification.forward': 'Trasforma coordinate e feature, aggrega i punti e restituisce logits e matrici.',
        'augment_pointcloud': 'Nel training originale aumenta i dati con rotazioni attorno a Z e rumore gaussiano.',
        'ModelNet40Dataset.__init__': 'Configura il dataset originale e il numero di punti richiesto dal modello.',
        'ModelNet40Dataset.__len__': 'Comunica al DataLoader quanti oggetti sono disponibili nello split.',
        'ModelNet40Dataset.__getitem__': 'Prepara un oggetto del training originale con campionamento e normalizzazione.',
        'orthogonality_loss': 'Penalizza la distanza di M Mᵀ dall’identità; non impone una rotazione esatta.',
        'train_epoch': 'Esegue un’epoca di training completo: augmentation, loss, gradienti e aggiornamento.',
        'evaluate': 'Misura loss e accuracy con rete in eval, senza aggiornamenti dei pesi.',
        'main': 'Avvia il training completo fornito dall’autore; l’app locale non lo esegue.',
    },
    'strumenti/download_resources.py': {
        'get_json': 'Legge i metadati Hugging Face necessari per fissare revisione e file da acquisire.',
        'main': 'Scarica le risorse selezionate e verifica dimensioni e hash prima di registrarne l’origine.',
    },
    'strumenti/test_inferenza.py': {
        'TestInferenza.setUpClass': 'Carica una volta pesi reali e campione test condivisi dalle verifiche.',
        'TestInferenza.test_preprocessing': 'Verifica forma, centro, raggio, ripetibilità e gestione degli input non validi.',
        'TestInferenza.test_inspection_matches_real_forward': 'Confronta le viste col forward reale e controlla l’invarianza agli stessi punti riordinati.',
        'before_classifier': 'Cattura l’ingresso reale di fc1 per confrontarlo col descrittore mostrato.',
    },
    'esperimento_horse/esperimento.py': {
        'sha': 'Calcola l’impronta SHA256 per controllare che dati e checkpoint siano invariati.',
        'split_horses': 'Deduplica file mesh identici e crea pool di training e test fisso disgiunti.',
        'old_examples': 'Seleziona oggetti bilanciati per classe, separando replay train e valutazione test.',
        'extract': 'Legge le feature 256D prima di fc3; il main ha già congelato il modello in eval.',
        'hook': 'Copia l’ingresso di fc3: è l’embedding dopo fc1/fc2, non il descrittore 1024D.',
        'metrics': 'Misura riconoscimento dei cavalli, falsi positivi e conservazione delle vecchie classi.',
        'fit_head': 'Ottimizza tutte le 41 righe della nuova testa; la testa originale resta separata.',
        'main': 'Orchestra test fisso, feature congelate e prove indipendenti per budget e seed.',
        'load_rows': 'Carica gli XYZ locali dei cavalli con pickle disabilitato.',
    },
    'esperimento_horse/scarica_cavalli.py': {
        'fetch': 'Controlla il limite di byte prima e durante il download del singolo file.',
        'main': 'Seleziona horse dalle annotazioni LVIS e registra le point cloud scaricate nel manifest.',
        'convert': 'Converte una scena GLB in triangoli, campiona la superficie e salva solo gli XYZ.',
    },
    'esperimento_horse/test_esperimento.py': {
        'ProtocolloTest.test_split_disgiunto_e_deduplicato': 'Controlla che file identici non finiscano sia nel supporto sia nel test.',
        'ProtocolloTest.test_un_esempio_non_modifica_originale': 'Verifica one-shot, ripetibilità e mancata modifica della testa di partenza.',
        'ProtocolloTest.test_rispondere_sempre_horse_non_basta': 'Dimostra che la sola recall non basta per valutare la nuova classe.',
    },
}


def main():
    # Inserimenti dall’ultima riga alla prima: le posizioni originali restano valide.
    count = 0
    for filename, descriptions in COMMENTS.items():
        path = ROOT / filename
        original = path.read_text(encoding='utf-8')
        tree = ast.parse(original)
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        lines = original.splitlines(keepends=True)
        insertions = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            parent = parents.get(node)
            key = f'{parent.name}.{node.name}' if isinstance(parent, ast.ClassDef) else node.name
            comment = descriptions.get(key)
            if not comment:
                continue
            marker = '# Scopo: ' + comment
            if marker in original:
                continue
            row = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            insertions.append((row, ' ' * node.col_offset + marker + '\n'))
        for row, text in sorted(insertions, reverse=True):
            lines.insert(row, text)
        updated = ''.join(lines)
        # Il confronto ignora i commenti e le posizioni: tutte le istruzioni restano identiche.
        assert ast.dump(tree) == ast.dump(ast.parse(updated)), filename
        path.write_text(updated, encoding='utf-8')
        count += len(insertions)
    print(f'Aggiunti {count} commenti; AST dei sorgenti invariato.')


if __name__ == '__main__':
    main()
