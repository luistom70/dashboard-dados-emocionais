import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io 
from utils import carregar_dados, definir_quadrante

st.set_page_config(layout="wide")
st.title("👤 Individual Athlete Analysis (FaceReader)")

# Verifica se o ficheiro foi carregado na Home
if "arquivo" not in st.session_state:
    st.warning("Please upload the file on the Home page first.")
    st.stop()

# Carregar dados
df = carregar_dados(st.session_state["arquivo"])
df["ID"] = df["Atleta"].astype("category").cat.codes + 1  # criar identificador

atletas = df[["Atleta", "ID"]].drop_duplicates().sort_values("ID")
mapa_ids = dict(zip(atletas["Atleta"], atletas["ID"]))

# Seleção de Atleta
atleta_selecionado = st.selectbox("Select Athlete for Analysis:", sorted(mapa_ids, key=mapa_ids.get))
id_atleta = mapa_ids[atleta_selecionado]
df_atleta = df[df["Atleta"] == atleta_selecionado]

# --- 1. VARIAÇÃO DAS EMOÇÕES ---
st.subheader(f"📈 Basic Emotions Variation - ID {id_atleta}")

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
    
    # Ordenar intervalos corretamente
    intervalos = sorted(df_atleta.index.unique(), key=lambda x: int(x.split("_")[1]))
    df_atleta = df_atleta.loc[intervalos]

    fig, ax = plt.subplots(figsize=(12, 5))
    for emotion in emotion_cols:
        ax.plot(df_atleta.index, df_atleta[emotion], marker='o', label=emotion)

    ax.set_title(f"Emotion Variation by Interval - ID {atleta_id}")
    ax.set_xlabel("Interval (Stimulus)")
    ax.set_ylabel("Variation (Max - Min)")
    
    # Ajustar labels do eixo X para "Image 1", "Image 2", etc.
    labels_clean = [label.replace("Intervalo_", "Image ") for label in df_atleta.index]
    ax.set_xticks(range(len(df_atleta.index)))
    ax.set_xticklabels(labels_clean, rotation=45)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    st.pyplot(fig)
    
    # Botão de Download
    fn = io.BytesIO()
    fig.savefig(fn, format='png', bbox_inches='tight')
    st.download_button(
        label="💾 Download Emotion Variation Graph",
        data=fn,
        file_name=f"emotion_variation_ID{atleta_id}.png",
        mime="image/png"
    )

variacao = calcular_variacao_emocoes(df)
plotar_variacao_emocoes(variacao, id_atleta)


# --- 2. VALENCE E AROUSAL NO TEMPO ---
st.subheader("📉 Valence and Arousal Over Time")
fig_time, ax_time = plt.subplots(figsize=(10, 4))
ax_time.plot(df_atleta["Video Time"], df_atleta["Valence"], label='Valence', color='blue', alpha=0.7)
ax_time.plot(df_atleta["Video Time"], df_atleta["Arousal"], label='Arousal', color='orange', alpha=0.7)
ax_time.set_title(f"Valence and Arousal Time Series - ID {id_atleta}")
ax_time.set_xlabel("Video Time (s)")
ax_time.set_ylabel("Intensity")
ax_time.legend()
ax_time.grid(True, alpha=0.3)
st.pyplot(fig_time)

# Botão de Download
fn_time = io.BytesIO()
fig_time.savefig(fn_time, format='png', bbox_inches='tight')
st.download_button(
    label="💾 Download Time Series Graph",
    data=fn_time,
    file_name=f"time_series_ID{id_atleta}.png",
    mime="image/png"
)

# --- 3. RADAR CHART (CORRIGIDO) ---
st.subheader("🧭 Average Emotional Profile (Expressive Only)")

# 1. Definir apenas as emoções expressivas (SEM NEUTRAL)
expressive_cols = ["Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted"]
medias = df_atleta[expressive_cols].mean()

# 2. Verificar se há dados expressivos para evitar divisão por zero
soma_expressiva = medias.sum()

if soma_expressiva > 0.001:  # Se houver alguma expressão mínima
    # Normalizar: Transforma em "Quota de Mercado" da emoção (Soma = 100%)
    valores_norm = medias / soma_expressiva
    
    # Preparar dados para o plot
    valores = valores_norm.values
    angles = np.linspace(0, 2 * np.pi, len(expressive_cols), endpoint=False).tolist()
    
    # Fechar o ciclo do radar (repetir o primeiro valor no fim)
    valores = np.concatenate((valores, [valores[0]]))
    angles += [angles[0]]
    
    # Plotar
    fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax_radar.plot(angles, valores, 'o-', linewidth=2, color='dodgerblue')
    ax_radar.fill(angles, valores, alpha=0.25, color='dodgerblue')
    
    # Ajustar labels
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), expressive_cols)
    ax_radar.set_title(f"Expressive Profile (Normalized) - ID {id_atleta}", y=1.1)
    
    # Definir limite fixo para ser fácil comparar (0 a 1, ou seja, 0% a 100%)
    ax_radar.set_ylim(0, 1)
    
    # Remover labels radiais numéricas para limpar o visual (opcional)
    # ax_radar.set_yticklabels([]) 
    
    ax_radar.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_radar)
    
    # Botão Download
    fn_radar = io.BytesIO()
    fig_radar.savefig(fn_radar, format='png', bbox_inches='tight')
    st.download_button(
        label="💾 Download Radar Chart",
        data=fn_radar,
        file_name=f"radar_chart_ID{id_atleta}.png",
        mime="image/png"
    )

else:
    st.info("⚠️ This athlete showed almost 100% Neutrality (no expressive emotions detected).")

# --- 4. TABELAS DE DADOS ---
st.subheader("📋 Mean and Max Values per Image")
df_intervalado = df_atleta[df_atleta["Intervalo"] != "Fora_Intervalo"]

# Calcular Médias e Máximos
medias = df_intervalado.groupby("Intervalo")[["Valence", "Arousal"]].mean().reset_index()
medias.columns = ["Intervalo", "Valence_Mean", "Arousal_Mean"]
maximos = df_intervalado.groupby("Intervalo")[["Valence", "Arousal"]].max().reset_index()
maximos.columns = ["Intervalo", "Valence_Max", "Arousal_Max"]

tabela = pd.merge(medias, maximos, on="Intervalo")
# Ordenar por número da imagem para ficar bonito
tabela["Order"] = tabela["Intervalo"].apply(lambda x: int(x.split('_')[1]))
tabela = tabela.sort_values("Order")
tabela["Image"] = tabela["Intervalo"].apply(lambda x: f"Image {int(x.split('_')[1])}")

tabela_final = tabela[["Image", "Valence_Mean", "Arousal_Mean", "Valence_Max", "Arousal_Max"]]

st.dataframe(tabela_final.style.format({
    "Valence_Mean": "{:.3f}",
    "Arousal_Mean": "{:.3f}",
    "Valence_Max": "{:.3f}",
    "Arousal_Max": "{:.3f}"
}))

# --- 5. QUADRANTES (SCATTER PLOTS) ---
st.subheader("📊 Valence/Arousal Quadrant Distribution")

col1, col2 = st.columns(2)

# --- Gráfico 1: Médias ---
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    # Quadrantes
    ax1.axvspan(-1, 0, ymin=0.5, ymax=1, facecolor='lightcoral', alpha=0.2) # Q2
    ax1.axvspan(0, 1, ymin=0.5, ymax=1, facecolor='khaki', alpha=0.2)      # Q1
    ax1.axvspan(-1, 0, ymin=0, ymax=0.5, facecolor='lightblue', alpha=0.2) # Q3
    ax1.axvspan(0, 1, ymin=0, ymax=0.5, facecolor='lightgreen', alpha=0.2) # Q4
    
    ax1.scatter(tabela["Valence_Mean"], tabela["Arousal_Mean"], s=100, color='mediumblue', edgecolors='white')
    
    # Anotar números das imagens
    for _, row in tabela.iterrows():
        img_num = row["Image"].split(" ")[1]
        ax1.text(row["Valence_Mean"] + 0.02, row["Arousal_Mean"], img_num, fontsize=9, fontweight='bold')
        
    ax1.axvline(0, color='black', linewidth=1)
    ax1.axhline(0.5, color='black', linewidth=1) # Arousal center is usually 0.5 in normalized [0,1]
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Mean Values - ID {id_atleta}")
    ax1.set_xlabel("Valence")
    ax1.set_ylabel("Arousal")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig1)
    
    # Download Botão 1
    fn_q1 = io.BytesIO()
    fig1.savefig(fn_q1, format='png', bbox_inches='tight')
    st.download_button(
        label="💾 Download Mean Quadrant Plot",
        data=fn_q1,
        file_name=f"quadrant_mean_ID{id_atleta}.png",
        mime="image/png"
    )

# --- Gráfico 2: Máximos ---
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    # Quadrantes
    ax2.axvspan(-1, 0, ymin=0.5, ymax=1, facecolor='lightcoral', alpha=0.2)
    ax2.axvspan(0, 1, ymin=0.5, ymax=1, facecolor='khaki', alpha=0.2)
    ax2.axvspan(-1, 0, ymin=0, ymax=0.5, facecolor='lightblue', alpha=0.2)
    ax2.axvspan(0, 1, ymin=0, ymax=0.5, facecolor='lightgreen', alpha=0.2)
    
    ax2.scatter(tabela["Valence_Max"], tabela["Arousal_Max"], s=100, color='darkred', edgecolors='white')
    
    for _, row in tabela.iterrows():
        img_num = row["Image"].split(" ")[1]
        ax2.text(row["Valence_Max"] + 0.02, row["Arousal_Max"], img_num, fontsize=9, fontweight='bold')
        
    ax2.axvline(0, color='black', linewidth=1)
    ax2.axhline(0.5, color='black', linewidth=1)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title(f"Max Values - ID {id_atleta}")
    ax2.set_xlabel("Valence")
    ax2.set_ylabel("Arousal")
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig2)

    # Download Botão 2
    fn_q2 = io.BytesIO()
    fig2.savefig(fn_q2, format='png', bbox_inches='tight')
    st.download_button(
        label="💾 Download Max Quadrant Plot",
        data=fn_q2,
        file_name=f"quadrant_max_ID{id_atleta}.png",
        mime="image/png"
    )

