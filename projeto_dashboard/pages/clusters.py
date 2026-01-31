import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import (
    carregar_dados, carregar_inqueritos, carregar_oasis,
    comparar_face_reader_vs_inquerito, adicionar_distancia_emocional,
    normalizar_oasis, calcular_distancias_ao_oasis
)

st.set_page_config(layout="wide")
st.title("🧠 Emotional Profile Clusters")

arquivo = st.session_state.get("arquivo")
if not arquivo:
    st.warning("Please upload the file on the Home page first.")
    st.stop()

# --- Preparar os dados
df = carregar_dados(arquivo)
df["ID"] = df["Atleta"].astype("category").cat.codes + 1

df_inq = carregar_inqueritos(arquivo)
df_oasis = carregar_oasis(arquivo)
df_oasis = normalizar_oasis(df_oasis)

df_comparado = comparar_face_reader_vs_inquerito(df, df_inq)
df_comparado = adicionar_distancia_emocional(df_comparado)

df_comparado["Imagem"] = df_comparado["Imagem"].astype(str).str.strip()
df_oasis["Imagem"] = df_oasis["Imagem"].astype(str).str.strip()
df_comparado = df_comparado.merge(df_oasis, on="Imagem", how="left")
df_comparado = calcular_distancias_ao_oasis(df_comparado)

# --- Clustering Analysis
st.subheader("📊 Athlete Clustering based on Average Emotional Profile")

# 1. Agrupar por ID (Perfil Médio)
df_comparado = df_comparado.merge(df[["Atleta", "ID"]].drop_duplicates(), left_on="Jogadora", right_on="Atleta")

df_cluster = (
    df_comparado
    .groupby("ID")[["Valence", "Arousal"]]
    .mean()
    .reset_index()
)
df_cluster["Label"] = df_cluster["ID"].apply(lambda x: f"ID {x}")

# 2. Configuração
k = st.slider("Select number of clusters (k)", min_value=2, max_value=5, value=3)

# 3. Aplicar K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
X = df_cluster[["Valence", "Arousal"]]
df_cluster["Cluster_Num"] = kmeans.fit_predict(X)
df_cluster["Cluster"] = df_cluster["Cluster_Num"].apply(lambda x: f"Cluster {x + 1}")

# 4. Calcular Silhouette Score
score = silhouette_score(X, df_cluster["Cluster_Num"])

# 5. Criar Gráfico (Scientific Style com Plotly Graph Objects)
fig = go.Figure()

# --- A. ADICIONAR QUADRANTES DE FUNDO (Contexto Semântico) ---
# Isto ajuda o leitor a saber o que significa "Cluster 1" ou "Cluster 2"
shapes = [
    # Distress (Canto Sup. Esq) - Vermelho
    dict(type="rect", x0=-1, y0=0.5, x1=0, y1=1, fillcolor="red", opacity=0.05, line_width=0),
    # Excitement (Canto Sup. Dir) - Verde
    dict(type="rect", x0=0, y0=0.5, x1=1, y1=1, fillcolor="green", opacity=0.05, line_width=0),
    # Depression/Boredom (Canto Inf. Esq) - Azul
    dict(type="rect", x0=-1, y0=0, x1=0, y1=0.5, fillcolor="blue", opacity=0.05, line_width=0),
    # Relaxation (Canto Inf. Dir) - Laranja
    dict(type="rect", x0=0, y0=0, x1=1, y1=0.5, fillcolor="orange", opacity=0.05, line_width=0)
]
fig.update_layout(shapes=shapes)

# --- B. ADICIONAR PONTOS (CLUSTERS) ---
# Adicionar um trace por cluster para ter cores diferentes e legenda automática
colors = px.colors.qualitative.Bold # Paleta de cores fortes e distintas
for i in range(k):
    cluster_data = df_cluster[df_cluster["Cluster_Num"] == i]
    fig.add_trace(go.Scatter(
        x=cluster_data["Valence"],
        y=cluster_data["Arousal"],
        mode='markers+text',
        name=f'Cluster {i+1}',
        text=cluster_data["Label"], # ID do atleta
        textposition="top center",
        textfont=dict(size=14, color='black', family="Arial Black"), # Texto legível
        marker=dict(
            size=18, # PONTOS GRANDES
            color=colors[i % len(colors)],
            line=dict(width=2, color='black'), # Borda preta para contraste
            symbol='circle'
        )
    ))

# --- C. ADICIONAR CENTRÓIDES ---
centroids = kmeans.cluster_centers_
fig.add_trace(go.Scatter(
    x=centroids[:, 0],
    y=centroids[:, 1],
    mode='markers',
    name='Centroids',
    marker=dict(symbol='x', size=20, color='black', line=dict(width=4)),
    showlegend=True
))

# --- D. LAYOUT CIENTÍFICO (Letras Grandes e Fundo Branco) ---
fig.update_layout(
    title=dict(
        text=f"Emotional Profile Clusters (k={k})",
        font=dict(size=24, family="Arial", color="black"),
        y=0.95
    ),
    xaxis=dict(
        title="Mean Valence (Negative ↔ Positive)",
        title_font=dict(size=20, family="Arial Black"),
        tickfont=dict(size=16),
        range=[-1.1, 1.1], # Margem para não cortar
        showgrid=True,
        gridcolor='lightgrey',
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=2
    ),
    yaxis=dict(
        title="Mean Arousal (Low ↔ High)",
        title_font=dict(size=20, family="Arial Black"),
        tickfont=dict(size=16),
        range=[-0.1, 1.1],
        showgrid=True,
        gridcolor='lightgrey'
    ),
    legend=dict(
        font=dict(size=16),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        yanchor="top", y=0.99, xanchor="left", x=0.01
    ),
    plot_bgcolor='white', # Fundo branco absoluto
    height=700,
    margin=dict(l=80, r=40, t=80, b=80)
)

# --- E. ANOTAÇÕES (SCORE E SIGNIFICADO) ---
# Silhouette Score
fig.add_annotation(
    text=f"<b>Silhouette Score: {score:.3f}</b>",
    xref="paper", yref="paper",
    x=1, y=1.08, # Canto superior direito (fora do gráfico)
    showarrow=False,
    font=dict(size=16, color="black"),
    bgcolor="white", bordercolor="black", borderwidth=1
)

# Etiquetas Semânticas (Discretas)
fig.add_annotation(x=-0.9, y=0.95, text="DISTRESS", showarrow=False, font=dict(size=14, color="gray"))
fig.add_annotation(x=0.9, y=0.95, text="EXCITEMENT", showarrow=False, font=dict(size=14, color="gray"))
fig.add_annotation(x=-0.9, y=0.05, text="BOREDOM", showarrow=False, font=dict(size=14, color="gray"))
fig.add_annotation(x=0.9, y=0.05, text="RELAXATION", showarrow=False, font=dict(size=14, color="gray"))

# Configurar Download Alta Resolução
config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'kmeans_clusters_scientific',
        'height': 1000, # Alta resolução
        'width': 1200,
        'scale': 2
    }
}

st.plotly_chart(fig, use_container_width=True, config=config)

# Tabela
st.subheader("📋 Athletes per Cluster")
df_tabela = df_cluster[["Label", "Cluster", "Valence", "Arousal"]].rename(columns={"Label": "Athlete ID"})
st.dataframe(df_tabela.sort_values("Cluster").style.format({
    "Valence": "{:.3f}",
    "Arousal": "{:.3f}"
}))



