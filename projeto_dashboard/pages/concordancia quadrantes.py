import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils import (
    carregar_dados, carregar_inqueritos,
    definir_quadrante, comparar_face_reader_vs_inquerito
)

st.set_page_config(layout="wide")
st.title("🧭 Concordância por Quadrantes Emocionais")

if "arquivo" not in st.session_state:
    st.warning("Por favor, carrega o ficheiro na página Home.")
    st.stop()

# Carregar dados
df = carregar_dados(st.session_state["arquivo"])
df_inq = carregar_inqueritos(st.session_state["arquivo"])
df["ID"] = df["Atleta"].astype("category").cat.codes + 1
df_comparado = comparar_face_reader_vs_inquerito(df, df_inq)
df_comparado = df_comparado.merge(df[["Atleta", "ID"]].drop_duplicates(), left_on="Jogadora", right_on="Atleta")


# Atribuir quadrantes
df_comparado["Quadrante_FaceReader"] = df_comparado.apply(lambda row: definir_quadrante(row["Valence"], row["Arousal"]), axis=1)
df_comparado["Quadrante_Inquerito"] = df_comparado.apply(lambda row: definir_quadrante(row["Valence_Inquerito"], row["Arousal_Inquerito"]), axis=1)
df_comparado["Concorda"] = df_comparado["Quadrante_FaceReader"] == df_comparado["Quadrante_Inquerito"]

# Concordância individual
st.subheader("📌 Concordância Individual por Jogadora")
id_sel = st.selectbox("Seleciona uma jogadora (ID):", sorted(df_comparado["ID"].unique()))
df_sel = df_comparado[df_comparado["ID"] == id_sel]


st.write(f"**Taxa de concordância para ID {id_sel}:** `{df_sel['Concorda'].mean() * 100:.1f}%`")
st.dataframe(df_sel[["Imagem", "Quadrante_FaceReader", "Quadrante_Inquerito", "Concorda"]])

# Concordância geral por jogadora
st.subheader("📊 Percentagem de Concordância por Jogadora")
concordancias = df_comparado.groupby(["Jogadora", "ID"])["Concorda"].mean().reset_index()
concordancias["% Concordância"] = concordancias["Concorda"] * 100

fig_conc = px.bar(
    concordancias,
    x="ID",
    y="% Concordância",
    title="Concordância de Quadrantes (FaceReader vs Inquérito)",
    text_auto=".1f",
    labels={"ID": "Jogadora (ID)", "% Concordância": "Concordância (%)"}
)
fig_conc.update_layout(yaxis_range=[0, 100])
st.plotly_chart(fig_conc, use_container_width=True)

# Heatmap geral
st.subheader("📌 Mapa de Quadrantes: Todos os Dados")
tabela_quadrantes = pd.crosstab(
    df_comparado["Quadrante_FaceReader"],
    df_comparado["Quadrante_Inquerito"]
)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(tabela_quadrantes, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
ax.set_title("Distribuição Cruzada dos Quadrantes")
ax.set_xlabel("Quadrante (Inquérito)")
ax.set_ylabel("Quadrante (FaceReader)")
st.pyplot(fig)
