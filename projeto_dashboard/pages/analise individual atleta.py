import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils import carregar_dados, definir_quadrante

st.set_page_config(layout="wide")
st.title("👤 Análise Individual da Atleta (FaceReader)")

# Verifica se o ficheiro foi carregado na Home
if "arquivo" not in st.session_state:
    st.warning("Por favor, carrega o ficheiro na página Home.")
    st.stop()

# Carregar dados
df = carregar_dados(st.session_state["arquivo"])
df["ID"] = df["Atleta"].astype("category").cat.codes + 1  # criar identificador

atletas = df[["Atleta", "ID"]].drop_duplicates().sort_values("ID")
mapa_ids = dict(zip(atletas["Atleta"], atletas["ID"]))
atleta_selecionado = st.selectbox("Seleciona a atleta para análise:", sorted(mapa_ids, key=mapa_ids.get))
id_atleta = mapa_ids[atleta_selecionado]
df_atleta = df[df["Atleta"] == atleta_selecionado]

st.subheader(f"📈 Variação das Emoções Básicas - ID {id_atleta}")

def calcular_variacao_emocoes(df):
    emotion_cols = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted", "Valence", "Arousal"]
    df_intervalado = df[df["Intervalo"] != "Fora_Intervalo"]
    variacao = df_intervalado.groupby(["Intervalo", "Atleta"])[emotion_cols].agg(lambda x: x.max() - x.min())
    return variacao.reset_index()

def plotar_variacao_emocoes(df_variacao, atleta_id):
    emotion_cols = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted"]
    df_variacao = df_variacao.copy()
    df_variacao["ID"] = df_variacao["Atleta"].map(mapa_ids)
    df_atleta = df_variacao[df_variacao["ID"] == atleta_id].set_index("Intervalo")
    intervalos = sorted(df_atleta.index.unique(), key=lambda x: int(x.split("_")[1]))
    df_atleta = df_atleta.loc[intervalos]

    fig, ax = plt.subplots(figsize=(12, 5))
    for emotion in emotion_cols:
        ax.plot(df_atleta.index, df_atleta[emotion], marker='o', label=emotion)

    ax.set_title(f"Variação das Emoções por Intervalo - ID {atleta_id}")
    ax.set_xlabel("Intervalo")
    ax.set_ylabel("Variação (Máx - Mín)")
    ax.set_xticks(range(len(df_atleta.index)))
    ax.set_xticklabels(df_atleta.index, rotation=45)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

variacao = calcular_variacao_emocoes(df)
plotar_variacao_emocoes(variacao, id_atleta)


# Linha de Valence e Arousal ao longo do tempo
st.subheader("📉 Valence e Arousal ao Longo do Tempo")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_atleta["Video Time"], df_atleta["Valence"], label='Valence', color='blue')
ax.plot(df_atleta["Video Time"], df_atleta["Arousal"], label='Arousal', color='orange')
ax.set_title("Valence e Arousal ao Longo do Tempo")
ax.set_xlabel("Tempo de Vídeo")
ax.set_ylabel("Valor")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Radar Emocional
st.subheader("🧭 Perfil Emocional Médio (Radar)")
emotion_cols = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted"]
medias = df_atleta[emotion_cols].mean()
expressivas = medias.drop("Neutral")
expressivas_norm = expressivas / expressivas.sum()
medias.update(expressivas_norm)
valores = medias.values
angles = np.linspace(0, 2 * np.pi, len(emotion_cols), endpoint=False).tolist()
valores = np.concatenate((valores, [valores[0]]))
angles += [angles[0]]
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, valores, 'o-', linewidth=2)
ax.fill(angles, valores, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), emotion_cols)
ax.set_title(f"Perfil Emocional (Normalizado) - ID {id_atleta}", y=1.1)


ax.grid(True)
st.pyplot(fig)

# Tabela com valores médios e máximos
st.subheader("📋 Valores Médios e Máximos por Imagem")
df_intervalado = df_atleta[df_atleta["Intervalo"] != "Fora_Intervalo"]
medias = df_intervalado.groupby("Intervalo")[["Valence", "Arousal"]].mean().reset_index()
medias.columns = ["Intervalo", "Valence_Medio", "Arousal_Medio"]
maximos = df_intervalado.groupby("Intervalo")[["Valence", "Arousal"]].max().reset_index()
maximos.columns = ["Intervalo", "Valence_Max", "Arousal_Max"]
tabela = pd.merge(medias, maximos, on="Intervalo")
tabela["Imagem"] = tabela["Intervalo"].apply(lambda x: f"Imagem {int(x.split('_')[1])}")
tabela = tabela[["Imagem", "Valence_Medio", "Arousal_Medio", "Valence_Max", "Arousal_Max"]]
st.dataframe(tabela.style.format({
    "Valence_Medio": "{:.3f}",
    "Arousal_Medio": "{:.3f}",
    "Valence_Max": "{:.3f}",
    "Arousal_Max": "{:.3f}"
}))

# Tabela com quadrantes por imagem
st.subheader("🧭 Quadrantes por Imagem")
tabela["Quadrante_Medio"] = tabela.apply(lambda row: definir_quadrante(row["Valence_Medio"], row["Arousal_Medio"]), axis=1)
tabela["Quadrante_Maximo"] = tabela.apply(lambda row: definir_quadrante(row["Valence_Max"], row["Arousal_Max"]), axis=1)
st.dataframe(tabela[["Imagem", "Quadrante_Medio", "Quadrante_Maximo"]])

# Gráficos com pontos nos quadrantes
st.subheader("📊 Distribuição Valence/Arousal nos Quadrantes")

# Médias
fig1, ax1 = plt.subplots(figsize=(7, 6))
ax1.axvspan(-1, 0, ymin=0.5, ymax=1, facecolor='lightcoral', alpha=0.3)
ax1.axvspan(0, 1, ymin=0.5, ymax=1, facecolor='khaki', alpha=0.3)
ax1.axvspan(-1, 0, ymin=0, ymax=0.5, facecolor='lightblue', alpha=0.3)
ax1.axvspan(0, 1, ymin=0, ymax=0.5, facecolor='lightgreen', alpha=0.3)
ax1.scatter(tabela["Valence_Medio"], tabela["Arousal_Medio"], s=100, color='mediumblue')
for _, row in tabela.iterrows():
    ax1.text(row["Valence_Medio"] + 0.01, row["Arousal_Medio"], row["Imagem"], fontsize=8)
ax1.axvline(0, color='black')
ax1.axhline(0.5, color='black')
ax1.set_xlim(-1, 1)
ax1.set_ylim(0, 1)
ax1.set_title(f"Valores Médios - ID {id_atleta}")
ax1.set_xlabel("Valence")
ax1.set_ylabel("Arousal")
ax1.grid(True)
st.pyplot(fig1)

# Máximos
fig2, ax2 = plt.subplots(figsize=(7, 6))
ax2.axvspan(-1, 0, ymin=0.5, ymax=1, facecolor='lightcoral', alpha=0.3)
ax2.axvspan(0, 1, ymin=0.5, ymax=1, facecolor='khaki', alpha=0.3)
ax2.axvspan(-1, 0, ymin=0, ymax=0.5, facecolor='lightblue', alpha=0.3)
ax2.axvspan(0, 1, ymin=0, ymax=0.5, facecolor='lightgreen', alpha=0.3)
ax2.scatter(tabela["Valence_Max"], tabela["Arousal_Max"], s=100, color='darkred')
for _, row in tabela.iterrows():
    ax2.text(row["Valence_Max"] + 0.01, row["Arousal_Max"], row["Imagem"], fontsize=8)
ax2.axvline(0, color='black')
ax2.axhline(0.5, color='black')
ax2.set_xlim(-1, 1)
ax2.set_ylim(0, 1)
ax2.set_title(f"Valores Máximos - ID {id_atleta}")
ax2.set_xlabel("Valence")
ax2.set_ylabel("Arousal")
ax2.grid(True)
st.pyplot(fig2)
