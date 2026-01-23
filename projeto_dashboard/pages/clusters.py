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

# --- Clustering
st.subheader("📊 Athlete Clustering based on Average Emotional Profile")

# 1. Agrupar por ID (Perfil Médio da Atleta)
df_comparado = df_comparado.merge(df[["Atleta", "ID"]].drop_duplicates(), left_on="Jogadora", right_on="Atleta")

df_cluster = (
    df_comparado
    .groupby("ID")[["Valence", "Arousal"]]
    .mean()
    .reset_index()
)
df_cluster["Label"] = df_cluster["ID"].apply(lambda x: f"ID {x}")

# 2. Configuração do Modelo
col1, col2 = st.columns([1, 2])
with col1:
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=5, value=3)

# 3. Aplicar K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
X = df_cluster[["Valence", "Arousal"]]
df_cluster["Cluster_Num"] = kmeans.fit_predict(X)
df_cluster["Cluster"] = df_cluster["Cluster_Num"].apply(lambda x: f"Cluster {x + 1}")

# 4. Calcular Silhouette Score (Qualidade Matemática)
score = silhouette_score(X, df_cluster["Cluster_Num"])

# Mostrar métrica de qualidade
with col2:
    st.metric(
        label="Silhouette Score (Cluster Quality)", 
        value=f"{score:.3f}", 
        delta="Good > 0.4" if score > 0.4 else "Weak < 0.25",
        help="Values near 1.0 indicate excellent separation. Values near 0 indicate overlapping clusters."
    )

# 5. Criar Gráfico com Centróides
fig = px.scatter(
    df_cluster,
    x="Valence",
    y="Arousal",
    color="Cluster",
    text="Label",
    title=f"Emotional Profile Clusters (k={k})",
    labels={"Valence": "Mean Valence", "Arousal": "Mean Arousal"},
    height=600
)

# Adicionar Centróides (Os "Centros" dos Grupos)
centroids = kmeans.cluster_centers_
fig.add_trace(
    go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        marker=dict(symbol='x', size=15, color='black', line=dict(width=2)),
        name='Centroids',
        showlegend=False
    )
)

# Estilização Profissional
fig.update_traces(
    textposition="top center", 
    textfont=dict(color="black", size=12), 
    marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey'))
)

fig.update_layout(
    plot_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor='lightgrey', range=[-1.1, 1.1], zeroline=True, zerolinecolor='black'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey', range=[-0.1, 1.1], zeroline=True, zerolinecolor='black'),
    legend_title_text='Group'
)

# Configurar Download
config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'kmeans_clusters_high_res',
        'height': 800,
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


