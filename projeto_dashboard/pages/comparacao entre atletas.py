import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils import carregar_dados

st.set_page_config(layout="wide")
st.title("👥 Comparação Entre Atletas: Valence e Arousal")

# Verifica se o ficheiro foi carregado
if "arquivo" not in st.session_state:
    st.warning("Por favor, carrega o ficheiro na página Home.")
    st.stop()

# Carregar dados
df = carregar_dados(st.session_state["arquivo"])
df["ID"] = df["Atleta"].astype("category").cat.codes + 1  # criar identificador

st.subheader("📌 Comparação Geral entre Atletas (Interativo)")
intervalos_disponiveis = ["Geral"] + sorted(
    df["Intervalo"].unique(),
    key=lambda x: int(x.split("_")[1]) if x.startswith("Intervalo") else 999
)
intervalo_selecionado = st.selectbox("Seleciona um intervalo para comparar os atletas:", intervalos_disponiveis)

# Filtrar dados conforme seleção
if intervalo_selecionado != "Geral":
    df_filtrado = df[df["Intervalo"] == intervalo_selecionado]
else:
    df_filtrado = df[df["Intervalo"] != "Fora_Intervalo"]

# Calcular médias e máximos por atleta
medias = df_filtrado.groupby("ID")[["Valence", "Arousal"]].mean().reset_index()
maximos = df_filtrado.groupby("ID")[["Valence", "Arousal"]].max().reset_index()

# --- Gráfico de MÉDIAS ---
st.markdown("### 🎯 Perfil Emocional Médio")
fig_media = go.Figure()

fig_media.add_shape(type="rect", x0=-1, x1=0, y0=0.5, y1=1, fillcolor="lightcoral", opacity=0.3, line_width=0)
fig_media.add_shape(type="rect", x0=0, x1=1, y0=0.5, y1=1, fillcolor="khaki", opacity=0.3, line_width=0)
fig_media.add_shape(type="rect", x0=-1, x1=0, y0=0, y1=0.5, fillcolor="lightblue", opacity=0.3, line_width=0)
fig_media.add_shape(type="rect", x0=0, x1=1, y0=0, y1=0.5, fillcolor="lightgreen", opacity=0.3, line_width=0)

fig_media.add_trace(go.Scatter(
    x=medias["Valence"],
    y=medias["Arousal"],
    mode='markers+text',
    text=[f"ID {i}" for i in medias["ID"]],
    textposition="top center",
    textfont=dict(color='black'),
    marker=dict(size=12, color='indigo', line=dict(width=1, color='white')),
    hovertemplate="ID: %{text}<br>Valence Médio: %{x:.2f}<br>Arousal Médio: %{y:.2f}<extra></extra>"
))

fig_media.update_layout(
    title=f"Valence e Arousal Médios - {intervalo_selecionado if intervalo_selecionado != 'Geral' else 'Todos os Intervalos'}",
    xaxis=dict(title="Valence", range=[-1, 1]),
    yaxis=dict(title="Arousal", range=[0, 1]),
    height=500,
    showlegend=False,
    plot_bgcolor='white'
)

st.plotly_chart(fig_media, use_container_width=True)

# --- Gráfico de MÁXIMOS ---
st.markdown("### 🚀 Picos Emocionais (Valores Máximos)")
fig_max = go.Figure()

fig_max.add_shape(type="rect", x0=-1, x1=0, y0=0.5, y1=1, fillcolor="lightcoral", opacity=0.3, line_width=0)
fig_max.add_shape(type="rect", x0=0, x1=1, y0=0.5, y1=1, fillcolor="khaki", opacity=0.3, line_width=0)
fig_max.add_shape(type="rect", x0=-1, x1=0, y0=0, y1=0.5, fillcolor="lightblue", opacity=0.3, line_width=0)
fig_max.add_shape(type="rect", x0=0, x1=1, y0=0, y1=0.5, fillcolor="lightgreen", opacity=0.3, line_width=0)

fig_max.add_trace(go.Scatter(
    x=maximos["Valence"],
    y=maximos["Arousal"],
    mode='markers+text',
    text=[f"ID {i}" for i in maximos["ID"]],
    textposition="top center",
    textfont=dict(color='black'),
    marker=dict(size=12, color='crimson', line=dict(width=1, color='white')),
    hovertemplate="ID: %{text}<br>Valence Máximo: %{x:.2f}<br>Arousal Máximo: %{y:.2f}<extra></extra>"
))

fig_max.update_layout(
    title=f"Valence e Arousal Máximos - {intervalo_selecionado if intervalo_selecionado != 'Geral' else 'Todos os Intervalos'}",
    xaxis=dict(title="Valence", range=[-1, 1]),
    yaxis=dict(title="Arousal", range=[0, 1]),
    height=500,
    showlegend=False,
    plot_bgcolor='white'
)

st.plotly_chart(fig_max, use_container_width=True)

# --- Exportar dados ---
st.markdown("### 💾 Exportar Dados")
nome_base = intervalo_selecionado.replace(" ", "_").lower()
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "📥 Exportar Médias (CSV)",
        data=medias.to_csv(index=False).encode("utf-8"),
        file_name=f"medias_valence_arousal_{nome_base}.csv",
        mime="text/csv"
    )
with col2:
    st.download_button(
        "📥 Exportar Máximos (CSV)",
        data=maximos.to_csv(index=False).encode("utf-8"),
        file_name=f"maximos_valence_arousal_{nome_base}.csv",
        mime="text/csv"
    )

