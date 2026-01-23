import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

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
# Criar ID numérico
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

# Calcular média por atleta (Agrupando APENAS por ID para anonimizar)
# Garantimos que o ID existe no df_comparado
df_comparado = df_comparado.merge(df[["Atleta", "ID"]].drop_duplicates(), left_on="Jogadora", right_on="Atleta")

# Agrupar por ID (Removendo o nome 'Jogadora' da equação)
df_cluster = (
    df_comparado
    .groupby("ID")[["Valence", "Arousal"]]
    .mean()
    .reset_index()
)

# Criar Label Anónima (ex: "ID 1")
df_cluster["Label"] = df_cluster["ID"].apply(lambda x: f"ID {x}")

# Slider para número de clusters
k = st.slider("Select number of clusters (k)", min_value=2, max_value=6, value=3)

# Aplicar K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
df_cluster["Cluster_Num"] = kmeans.fit_predict(df_cluster[["Valence", "Arousal"]])
df_cluster["Cluster"] = df_cluster["Cluster_Num"].apply(lambda x: f"Cluster {x + 1}")

# Gráfico (Plotly)
fig = px.scatter(
    df_cluster,
    x="Valence",
    y="Arousal",
    color="Cluster",
    text="Label",  # Usa a Label ID em vez do nome
    title=f"Emotional Profile Clusters (K-Means, k={k})",
    labels={"Valence": "Mean Valence", "Arousal": "Mean Arousal"},
    height=600
)

# Melhorias visuais para o artigo
fig.update_traces(
    textposition="top center", 
    textfont=dict(color="black", size=12), 
    marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey'))
)

# Fundo branco e grelha cinza (estilo académico)
fig.update_layout(
    plot_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='black'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='black'),
    legend_title_text='Group'
)

# Configuração para Download em Alta Resolução
# Isto adiciona opções ao botão da câmara fotográfica no gráfico
config = {
    'toImageButtonOptions': {
        'format': 'png', # ou 'svg' para vetor
        'filename': 'kmeans_clusters_high_res',
        'height': 800,
        'width': 1200,
        'scale': 2 # Aumenta a resolução (escala 2x)
    }
}

st.plotly_chart(fig, use_container_width=True, config=config)

st.caption("ℹ️ To download the image for the paper: Hover over the chart and click the camera icon (📸) in the top right corner.")

# Tabela com IDs (Sem nomes)
st.subheader("📋 Athletes per Cluster")
df_tabela = df_cluster[["Label", "Cluster", "Valence", "Arousal"]].rename(columns={"Label": "Athlete ID"})
st.dataframe(df_tabela.sort_values("Cluster").style.format({
    "Valence": "{:.3f}",
    "Arousal": "{:.3f}"
}))


