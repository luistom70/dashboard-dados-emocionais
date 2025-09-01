import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import (
    carregar_dados, carregar_inqueritos, carregar_oasis,
    comparar_face_reader_vs_inquerito, adicionar_distancia_emocional,
    normalizar_oasis, calcular_distancias_ao_oasis
)

st.set_page_config(layout="wide")
st.title("📏 Distância ao OASIS")

if "arquivo" not in st.session_state:
    st.warning("Por favor, carrega o ficheiro na página Home.")
    st.stop()

# Carregamento de dados
df = carregar_dados(st.session_state["arquivo"])
df_inq = carregar_inqueritos(st.session_state["arquivo"])
df_oasis = carregar_oasis(st.session_state["arquivo"])
df_oasis = normalizar_oasis(df_oasis)

# Preparar comparação
df_comparado = comparar_face_reader_vs_inquerito(df, df_inq)
df_comparado = adicionar_distancia_emocional(df_comparado)
df_comparado["Imagem"] = df_comparado["Imagem"].astype(str).str.strip()
df_oasis["Imagem"] = df_oasis["Imagem"].astype(str).str.strip()
df_comparado = df_comparado.merge(df_oasis, on="Imagem", how="left")
df_comparado = calcular_distancias_ao_oasis(df_comparado)

# Distância média por jogadora
st.subheader("📊 Distância Média ao OASIS por Jogadora")
df_dist_jog = df_comparado.groupby("Jogadora")[["Dist_FR_OASIS", "Dist_INQ_OASIS"]].mean().reset_index()
fig_jog = go.Figure()
fig_jog.add_trace(go.Bar(
    x=df_dist_jog["Jogadora"],
    y=df_dist_jog["Dist_FR_OASIS"],
    name="FaceReader",
    marker_color="indianred"
))
fig_jog.add_trace(go.Bar(
    x=df_dist_jog["Jogadora"],
    y=df_dist_jog["Dist_INQ_OASIS"],
    name="Inquérito",
    marker_color="royalblue"
))
fig_jog.update_layout(
    barmode='group',
    xaxis_title="Jogadora",
    yaxis_title="Distância Euclidiana Média ao OASIS",
    height=500
)
st.plotly_chart(fig_jog, use_container_width=True)

# Distância média por imagem
st.subheader("📷 Distância Média ao OASIS por Imagem")
df_dist_img = df_comparado.groupby("Imagem")[["Dist_FR_OASIS", "Dist_INQ_OASIS"]].mean().reset_index()
df_dist_img["Num"] = df_dist_img["Imagem"].str.extract(r'(\d+)').astype(int)
df_dist_img = df_dist_img.sort_values("Num")
fig_img = go.Figure()
fig_img.add_trace(go.Bar(
    x=df_dist_img["Imagem"],
    y=df_dist_img["Dist_FR_OASIS"],
    name="FaceReader",
    marker_color="indianred"
))
fig_img.add_trace(go.Bar(
    x=df_dist_img["Imagem"],
    y=df_dist_img["Dist_INQ_OASIS"],
    name="Inquérito",
    marker_color="royalblue"
))
fig_img.update_layout(
    barmode='group',
    xaxis_title="Imagem",
    yaxis_title="Distância Euclidiana Média ao OASIS",
    height=500
)
st.plotly_chart(fig_img, use_container_width=True)

# Exportação
st.markdown("### 💾 Exportar Tabelas")
col1, col2 = st.columns(2)
with col1:
    st.download_button("📥 Exportar por Jogadora", data=df_dist_jog.to_csv(index=False).encode("utf-8"), file_name="distancia_jogadoras_oasis.csv", mime="text/csv")
with col2:
    st.download_button("📥 Exportar por Imagem", data=df_dist_img.to_csv(index=False).encode("utf-8"), file_name="distancia_imagens_oasis.csv", mime="text/csv")
