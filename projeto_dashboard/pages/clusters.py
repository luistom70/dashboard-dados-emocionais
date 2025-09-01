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
st.title("🧠 Clusters de Perfis Emocionais")

arquivo = st.session_state.get("arquivo")
if not arquivo:
    st.warning("Por favor, carrega o ficheiro na página Home.")
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
st.subheader("📊 Agrupamento de Atletas com base no Perfil Emocional Médio")

# Calcular média por atleta
df_comparado = df_comparado.merge(df[["Atleta", "ID"]].drop_duplicates(), left_on="Jogadora", right_on="Atleta")
df_cluster = df_comparado.groupby("ID")[["Valence", "Arousal"]].mean().reset_index()

# Slider para número de clusters
k = st.slider("Seleciona o número de clusters", min_value=2, max_value=6, value=3)

# Aplicar K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
df_cluster["Cluster"] = kmeans.fit_predict(df_cluster[["Valence", "Arousal"]])

# Nomear os clusters como "Grupo 1", "Grupo 2", ...
df_cluster["Grupo"] = df_cluster["Cluster"].apply(lambda x: f"Grupo {x + 1}")

# --- Gráfico interativo
fig = px.scatter(
    df_cluster,
    x="Valence",
    y="Arousal",
    color="Grupo",
    text=df_cluster["ID"].apply(lambda x: f"ID {x}"),
    title=f"Clusters de Perfis Emocionais (K-Means, k={k})",
    labels={"Valence": "Valence Médio", "Arousal": "Arousal Médio"},
    height=600
)
fig.update_traces(textposition="top center", textfont=dict(color="black"), marker=dict(size=10))
fig.update_layout(plot_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)

# --- Tabela com jogadoras por grupo
st.subheader("📋 Jogadoras por Grupo")
df_cluster = df_cluster[["ID", "Grupo"]].rename(columns={"ID": "Identificador"})
st.dataframe(df_cluster.sort_values("Grupo"))
