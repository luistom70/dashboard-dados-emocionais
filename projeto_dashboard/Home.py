import streamlit as st
from utils import carregar_dados
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Dashboard Emocional de Atletas")

# Botão informativo de navegação
if st.button("🔁 Ir para Comparação FR / Inquérito / OASIS"):
    st.info("Abre o menu lateral esquerdo e clica na página **'Comparacao_FR_INQ_OASIS'** para ver as comparações.")

# Botão para limpar base de dados
if st.button("🗑️ Limpar ficheiro carregado"):
    if "arquivo" in st.session_state:
        del st.session_state["arquivo"]
        st.success("Ficheiro removido. Recarrega um novo Excel para continuar.")
        st.stop()


# Upload com persistência
if "arquivo" not in st.session_state:
    arquivo = st.file_uploader("Carrega o ficheiro Excel da base de dados", type=["xlsx"], key="uploader_home")
    if arquivo:
        st.session_state["arquivo"] = arquivo
else:
    arquivo = st.session_state["arquivo"]

if "arquivo" not in st.session_state:
    st.warning("Por favor, carrega um ficheiro Excel válido.")
    st.stop()

# Só se arquivo existir:
df = carregar_dados(st.session_state["arquivo"])



def calcular_variacao_emocoes(df):
    emotion_cols = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted", "Valence", "Arousal"]
    df_intervalado = df[df["Intervalo"] != "Fora_Intervalo"]
    variacao = df_intervalado.groupby(["Intervalo", "Atleta"])[emotion_cols].agg(lambda x: x.max() - x.min())
    return variacao.reset_index()

def plotar_variacao_emocoes(variacao_df, atleta):
    emotion_cols = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted"]
    intervalos = sorted(variacao_df["Intervalo"].unique(), key=lambda x: int(x.split("_")[1]))
    df_atleta = variacao_df[variacao_df["Atleta"] == atleta].set_index("Intervalo").loc[intervalos]

    fig, ax = plt.subplots(figsize=(12, 5))
    for emotion in emotion_cols:
        ax.plot(df_atleta.index, df_atleta[emotion], marker='o', label=emotion)

    ax.set_title(f"Variação das Emoções por Intervalo - {atleta}")
    ax.set_xlabel("Intervalo")
    ax.set_ylabel("Variação (Máx - Mín)")
    ax.set_xticks(range(len(df_atleta.index)))
    ax.set_xticklabels(df_atleta.index, rotation=45)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

def plot_valence_arousal_linha(df_atleta):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_atleta["Video Time"], df_atleta["Valence"], label='Valence', color='blue')
    ax.plot(df_atleta["Video Time"], df_atleta["Arousal"], label='Arousal', color='orange')
    ax.set_title("Valence e Arousal ao Longo do Tempo")
    ax.set_xlabel("Tempo de Vídeo")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_variacao_por_intervalo(df_atleta):
    emotion_cols = ["Valence", "Arousal"]
    variacoes = df_atleta[df_atleta["Intervalo"] != "Fora_Intervalo"].groupby("Intervalo")[emotion_cols].agg(lambda x: x.max() - x.min())
    st.write("### Variação (máximo - mínimo) por Intervalo")
    st.bar_chart(variacoes)

def plot_scatter_perfil(df, intervalo_selecionado):
    if intervalo_selecionado != "Geral":
        df = df[df["Intervalo"] == intervalo_selecionado]
    else:
        df = df[df["Intervalo"] != "Fora_Intervalo"]
    medias = df.groupby("Atleta")[["Valence", "Arousal"]].mean().reset_index()
    fig = go.Figure()

    fig.add_shape(type="rect", x0=-1, x1=0, y0=0.5, y1=1, fillcolor="lightcoral", opacity=0.3, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=1, y0=0.5, y1=1, fillcolor="khaki", opacity=0.3, line_width=0)
    fig.add_shape(type="rect", x0=-1, x1=0, y0=0, y1=0.5, fillcolor="lightblue", opacity=0.3, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=0.5, fillcolor="lightgreen", opacity=0.3, line_width=0)

    fig.add_trace(go.Scatter(
        x=medias["Valence"],
        y=medias["Arousal"],
        mode='markers+text',
        text=medias["Atleta"],
        textposition="top center",
        marker=dict(size=10, color="mediumblue")
    ))

    fig.update_layout(
        title=f"Perfil Emocional Médio - {intervalo_selecionado}",
        xaxis=dict(title="Valence", range=[-1, 1]),
        yaxis=dict(title="Arousal", range=[0, 1]),
        height=600,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_radar_emocoes(df_atleta, atleta):
    emotion_cols = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted"]
    medias = df_atleta[emotion_cols].mean()

    # Normalizar sem o Neutral (opcional)
    expressivas = medias.drop("Neutral")
    expressivas_norm = expressivas / expressivas.sum()
    medias.update(expressivas_norm)

    # Preparar dados do radar
    valores = medias.values
    angles = np.linspace(0, 2 * np.pi, len(emotion_cols), endpoint=False).tolist()
    valores = np.concatenate((valores, [valores[0]]))
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, valores, 'o-', linewidth=2)
    ax.fill(angles, valores, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), emotion_cols)
    ax.set_title(f"Perfil Emocional (Normalizado) - {atleta}", y=1.1)
    ax.grid(True)
    st.pyplot(fig)
    

# Interface principal
arquivo = st.session_state["arquivo"]
if arquivo:
    df = carregar_dados(arquivo)
    atletas = df["Atleta"].unique()
    atleta_selecionado = st.selectbox("Seleciona a atleta para análise:", sorted(atletas))
    df_atleta = df[df["Atleta"] == atleta_selecionado]

    st.subheader(f"📈 Variação das Emoções Básicas - {atleta_selecionado}")
    variacao = calcular_variacao_emocoes(df)
    plotar_variacao_emocoes(variacao, atleta_selecionado)

    col1, col2 = st.columns(2)
    with col1:
        plot_valence_arousal_linha(df_atleta)
    with col2:
        plot_variacao_por_intervalo(df_atleta)

    st.divider()
    st.subheader("📌 Comparação Geral entre Atletas (Interativo)")
    intervalos_disponiveis = ["Geral"] + sorted(df["Intervalo"].unique(), key=lambda x: int(x.split("_")[1]) if x.startswith("Intervalo") else 999)
    intervalo_selecionado = st.selectbox("Seleciona um intervalo para comparar os atletas:", intervalos_disponiveis)
    plot_scatter_perfil(df, intervalo_selecionado)
    plot_radar_emocoes(df_atleta, atleta_selecionado)

