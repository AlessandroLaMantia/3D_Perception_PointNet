"""Interfaccia essenziale: scegli un oggetto e confronta la risposta di PointNet."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from inferenza import (CATEGORIE, analizza, carica_modello, leggi_campione,
                       leggi_labels, nome_classe)

st.set_page_config(page_title='3D Perception', layout='wide')


# Cache delle sole etichette: evita letture ripetute durante i rerun di Streamlit.
# Scopo: Memorizza le etichette per evitare letture durante ogni interazione con Streamlit.
@st.cache_data
def labels_cached(split):
    return leggi_labels(split)


# Cache del campione originale, separata dalla cache delle predizioni.
# Scopo: Memorizza geometria ed etichetta usando split e indice come chiave.
@st.cache_data
def campione_cached(split, indice):
    return leggi_campione(split, indice)


# Esegue inferenza una volta per oggetto; nessun aggiornamento dei pesi.
# Scopo: Memorizza l’analisi del singolo oggetto: l’app non aggiorna i pesi.
@st.cache_data
def risultato_cached(split, indice):
    # Seed fisso: risultati ripetibili senza un controllo tecnico nell'interfaccia.
    points, _ = campione_cached(split, indice)
    return analizza(carica_modello(), points, seed=42)


# Scatter 3D con proporzioni geometriche conservate; i colori opzionali descrivono i massimi.
# Scopo: Crea uno scatter XYZ ruotabile mantenendo le proporzioni della geometria.
def figura(points, title, colors=None):
    marker = dict(size=2.5, opacity=0.9, color='#36a9e1')
    if colors is not None:
        marker.update(color=colors, colorscale='Turbo', showscale=True,
                      colorbar=dict(title='Canali'))
    fig = go.Figure(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                               mode='markers', marker=marker))
    fig.update_layout(title=title, height=450, margin=dict(l=0, r=0, b=0, t=45),
                      scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    return fig


# Scopo: Formatta il punteggio senza arrotondare a zero le alternative molto piccole.
def percentuale(value):
    # Evita di far sembrare nulle tutte le alternative piccole ma positive.
    if 0 < value < 0.01:
        return '<0,01%'
    return f'{value:.2f}%'.replace('.', ',')


# Ordina la softmax e mostra solo le cinque alternative principali, senza rinormalizzarle.
# Scopo: Mostra le cinque softmax maggiori sulla scala comune 0–100 senza rinormalizzarle.
def grafico_categorie(probs):
    top = np.argsort(-probs, kind='stable')[:5]
    values = probs[top] * 100
    names = [CATEGORIE[i][1].capitalize() for i in top]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation='h',
        marker_color=['#168a80'] + ['#9aa9bc'] * 4,
        text=[percentuale(float(v)) for v in values], textposition='outside',
        cliponaxis=False,
        hovertemplate='%{y}: %{x:.4f}%<extra></extra>',
    ))
    # Scala comune 0..100 e prima categoria in alto: confronti leggibili e onesti.
    fig.update_layout(height=320, margin=dict(l=10, r=90, t=15, b=40),
                      xaxis=dict(range=[0, 100], ticksuffix='%', dtick=25,
                                 title=None, fixedrange=True),
                      yaxis=dict(autorange='reversed', fixedrange=True),
                      showlegend=False, bargap=0.35)
    return fig


st.title('3D Perception')
st.write('Scegli un oggetto, osservalo in 3D e scopri come lo classifica la rete.')
category_slot = st.sidebar.container()
with st.sidebar.expander('Opzioni del dataset'):
    split = st.selectbox('Dataset', ['test', 'train'])
    st.caption('Test: oggetti separati dal training. Cambiare dataset non addestra la rete.')
labels = labels_cached(split)
# Valutiamo l'esempio corrente di ogni categoria, non l'intera categoria.
# La cache evita di rieseguire le 40 inferenze a ogni interazione.
examples = {}
with st.spinner('Verifica degli esempi delle 40 categorie...'):
    for class_id in range(40):
        candidates = np.flatnonzero(labels == class_id).tolist()
        pos = st.session_state.get(f'esempio_{split}_{class_id}', 0)
        sample_index = candidates[pos]
        predicted = int(risultato_cached(split, sample_index)['probabilities'].argmax())
        examples[class_id] = ('🟢' if predicted == class_id else '🔴', pos + 1)
with category_slot:
    category = st.selectbox('Categoria da esplorare', list(range(40)), key='categoria',
                            format_func=lambda i: f'{examples[i][0]} {CATEGORIE[i][1].capitalize()} · esempio {examples[i][1]}')
    st.caption('🟢 Esempio riconosciuto · 🔴 Esempio sbagliato. Il colore riguarda questo oggetto, non tutta la categoria.')
indices = np.flatnonzero(labels == category).tolist()
# L'indice tecnico resta interno. L'utente sfoglia esempi della stessa categoria.
position_key = f'esempio_{split}_{category}'
if position_key not in st.session_state:
    st.session_state[position_key] = 0
# Scopo: Avanza ciclicamente fra gli oggetti della classe prima del nuovo rerun.
def cambia_esempio():
    # Il callback aggiorna l'esempio prima di ricalcolare i pallini del menu.
    st.session_state[position_key] = (st.session_state[position_key] + 1) % len(indices)

st.sidebar.button('Mostra un altro esempio', disabled=len(indices) < 2,
                  key='altro_esempio', on_click=cambia_esempio)
position = st.session_state[position_key]
indice = indices[position]
st.sidebar.caption(f'Esempio {position + 1} di {len(indices)} della categoria scelta.')
points, label = campione_cached(split, indice)
st.subheader(f'{CATEGORIE[label][1].capitalize()} · Esempio {position + 1}')
st.caption('Trascina per ruotare; usa la rotella per ingrandire.')
st.plotly_chart(figura(points, 'Oggetto originale'), width='stretch')

# Cambiare oggetto invalida il risultato mostrato, evitando risposte obsolete.
selection = (split, indice)
if st.button('Invia alla rete', type='primary', key='inferisci'):
    st.session_state['analizzata'] = selection

if st.session_state.get('analizzata') == selection:
    with st.spinner('La rete sta analizzando i punti...'):
        result = risultato_cached(*selection)
    probs = result['probabilities']
    prediction = int(probs.argmax())
    st.subheader(f'La rete riconosce: {CATEGORIE[prediction][1]}')
    if prediction == label:
        st.success('La predizione coincide con la categoria del dataset.')
    else:
        st.warning(f'La predizione non coincide: nel dataset questo oggetto è {CATEGORIE[label][1]}.')
    st.markdown('**Le 5 categorie con punteggio maggiore**')
    st.plotly_chart(grafico_categorie(probs), width='stretch', config={'displayModeBar': False})
    st.caption('I punteggi confrontano le 40 categorie e sommano al 100% considerando tutte le classi. '
               'Un punteggio alto non garantisce una risposta corretta; le percentuali sono arrotondate.')

    # Dettagli didattici facoltativi: non necessari per leggere la classificazione.
    with st.expander('Approfondimento facoltativo: come elabora i punti'):
        st.write('La rete usa 1.024 punti centrati e normalizzati. La T-Net ne trasforma '
                 'le coordinate per la successiva elaborazione; non ricostruisce un nuovo oggetto.')
        left, right = st.columns(2)
        with left:
            st.plotly_chart(figura(result['input'], 'Punti dati alla rete'), width='stretch')
        with right:
            st.plotly_chart(figura(result['aligned'], 'Dopo la trasformazione interna'), width='stretch')
        st.caption('La trasformazione appresa può anche cambiare scala e forma: non è necessariamente una rotazione.')
        st.markdown('**Come la rete riassume i punti**')
        st.write('Per ogni punto calcola 1.024 caratteristiche. Il max pooling conserva '
                 'il massimo di ciascuna caratteristica, producendo un solo vettore per tutto l’oggetto.')
        st.plotly_chart(figura(result['input'], 'Contributi al max pooling', result['max_counts']), width='stretch')
        st.caption('Il colore conta in quanti canali attivi il punto fornisce il massimo. '
                   'Non indica parti dell’oggetto o importanza causale; a parità viene contato il primo punto.')
        st.write('Questo vettore di 1.024 valori è il descrittore interno: serve al classificatore '
                 'per scegliere la categoria. Non è un’immagine; mostrarlo come griglia non aiuta a leggere la risposta.')
