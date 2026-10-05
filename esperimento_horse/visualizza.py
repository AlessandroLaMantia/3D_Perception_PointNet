"""Interfaccia separata per presentare le misure del fine-tuning.

Questa dashboard e' pensata per mostrare chiaramente il comportamento del modello dopo
il fine-tuning few-shot su horse. L'obiettivo e' rendere comprensibili i risultati senza
fare confusione tra esperimenti, seed e dati di training/test.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title='3D Perception - Esperimento horse', layout='wide')
st.title('Quanti esempi servono per imparare horse?')
st.write('Fine-tuning del solo ultimo livello di PointNet: 40 categorie originali più cavallo.')
# Seleziona l ultima esecuzione con misure salvate: la dashboard non riaddestra la rete.
runs = sorted((HERE / 'risultati').glob('*/misure.csv'), reverse=True)
if not runs:
    st.info('Nessuna misura ancora disponibile. Esegui prima esperimento_horse/esperimento.py.')
    st.stop()
run = runs[0].parent
# Il protocollo stabilisce l'ordine degli oggetti nei file delle predizioni salvate.
# Le righe iniziali sono ModelNet40; i cavalli seguono nello stesso ordine degli ID.
protocol = json.loads((run / 'protocollo.json').read_text())
df = pd.read_csv(run / 'misure.csv')

st.subheader('Un cavallo del test: mai usato per il fine-tuning')
horse = st.selectbox('Oggetto di test', protocol['horse_test_ids'])
budget = st.selectbox('Numero di esempi usati', sorted(df.shots.unique().tolist()))

# La vista del singolo cavallo usa il primo seed; il grafico aggrega tutte le ripetizioni.
seed = sorted(df.seed.unique().tolist())[0]
points = np.load(HERE / 'dati' / f'{horse}.npy', allow_pickle=False)
cloud = go.Figure(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                              mode='markers', marker=dict(size=2, color='#36a9e1')))
cloud.update_layout(scene=dict(aspectmode='data'), height=430, margin=dict(l=0, r=0, t=0, b=0))

# Legge una predizione gia salvata per questo oggetto, budget e seed.
if st.button('Invia', type='primary'):
    predictions = np.load(run / f'predizioni_{budget}_{seed}.npz', allow_pickle=False)
    # Offset dei vecchi oggetti: la posizione nel file non coincide con l'indice del cavallo.
    index = len(protocol['old_test_ids']) + protocol['horse_test_ids'].index(horse)
    prediction = int(predictions['pred'][index])
    import sys
    sys.path.insert(0, str(HERE.parent))
    from inferenza import CATEGORIE
    name = 'cavallo (horse)' if prediction == 40 else CATEGORIE[prediction][1]
    st.markdown(f'**Predizione:** {name} · **Categoria di riferimento:** cavallo')
    st.plotly_chart(cloud, use_container_width=True)

else:
    st.markdown('**Predizione:** nessuna ancora · **Categoria di riferimento:** cavallo')
    st.plotly_chart(cloud, use_container_width=True)

# Media e deviazione standard tra seed sullo stesso test; non intervallo di confidenza.
fig = go.Figure()
for metric, name in [('horse_recall', 'Cavalli riconosciuti'), ('old_accuracy', 'Accuracy vecchie classi'),
                     ('accuracy', 'Accuracy complessiva')]:
    grouped = df.groupby('shots')[metric]
    means = grouped.mean() * 100
    stds = grouped.agg(lambda x: np.std(x, ddof=0)) * 100
    fig.add_trace(go.Scatter(x=means.index, y=means.values, name=name, mode='lines+markers',
                            line=dict(width=3), marker=dict(size=9),
                            error_y=dict(type='data', array=stds.values, visible=True)))
fig.update_layout(
    title='Accuracy in funzione del numero di esempi di cavallo',
    xaxis=dict(title='Esempi di cavallo nel training', type='category', tickmode='array', tickvals=sorted(df.shots.unique().tolist())),
    yaxis=dict(title='Percentuale', range=[0, 100]),
    template='plotly_white',
    margin=dict(l=30, r=160, t=90, b=100),
    legend=dict(orientation='v', x=1.02, y=1.0, xanchor='left', yanchor='top'),
    height=560,
)
st.plotly_chart(fig, use_container_width=True)
