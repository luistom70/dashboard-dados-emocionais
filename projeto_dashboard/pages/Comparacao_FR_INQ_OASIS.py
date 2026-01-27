import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import (
    carregar_dados, carregar_inqueritos, carregar_oasis,
    comparar_face_reader_vs_inquerito, adicionar_distancia_emocional,
    normalizar_oasis
)

st.set_page_config(layout="wide")
st.title("⚖️ Source Comparison: FaceReader vs. Survey vs. OASIS")

# Interface principal
arquivo = st.session_state.get("arquivo")
if not arquivo:
    st.warning("Please upload the file on the Home page first.")
    st.stop()

# --- Carregar Dados ---
df = carregar_dados(arquivo)
df_inq = carregar_inqueritos(arquivo)
df_oasis = carregar_oasis(arquivo)
df_oasis = normalizar_oasis(df_oasis)

# Processamento
df_comparado = comparar_face_reader_vs_inquerito(df, df_inq)
df_comparado = adicionar_distancia_emocional(df_comparado)

df_comparado["Imagem"] = df_comparado["Imagem"].astype(str).str.strip()
df_oasis["Imagem"] = df_oasis["Imagem"].astype(str).str.strip()
# Juntar OASIS
df_final = df_comparado.merge(df_oasis, on="Imagem", how="left")

# IDs para seleção individual
df["ID"] = df["Atleta"].astype("category").cat.codes + 1
df_final = df_final.merge(df[["Atleta", "ID"]].drop_duplicates(), left_on="Jogadora", right_on="Atleta")

# --- CRIAÇÃO DAS ABAS ---
tab_global, tab_individual = st.tabs(["🌍 Global Analysis (Paper)", "👤 Individual Analysis"])

# ==============================================================================
# ABA 1: ANÁLISE GLOBAL (Para o Artigo)
# ==============================================================================
with tab_global:
    st.header("Global Trends (All Athletes Aggregated)")
    st.caption("This view compares the mean values across all athletes to identify general trends and discrepancies.")

    # Calcular Médias Globais por Imagem
    df_global = df_final.groupby("Imagem")[
        ["Valence_FaceReader", "Valence_Inquerito", "Valence_OASIS",
         "Arousal_FaceReader", "Arousal_Inquerito", "Arousal_OASIS"]
    ].mean().reset_index()

    # Ordenar por número da imagem
    df_global["ImgNum"] = df_global["Imagem"].apply(lambda x: int(x.split("_")[1]))
    df_global = df_global.sort_values("ImgNum")
    df_global["Label"] = df_global["ImgNum"].apply(lambda x: f"Img {x}")

    # Seletor de Métrica
    metric = st.radio("Select Metric for Global Comparison:", ["Valence", "Arousal"], horizontal=True)

    # Preparar dados Plotly
    cols = [f"{metric}_FaceReader", f"{metric}_Inquerito", f"{metric}_OASIS"]
    df_melt = df_global.melt(id_vars=["Label"], value_vars=cols, var_name="Source", value_name="Value")
    df_melt["Source"] = df_melt["Source"].str.replace(f"{metric}_", "").replace("Inquerito", "Survey")

    # Cores
    colors = {"FaceReader": "#1f77b4", "Survey": "#ff7f0e", "OASIS": "#2ca02c"}

    fig_global = px.bar(
        df_melt,
        x="Label",
        y="Value",
        color="Source",
        barmode="group",
        color_discrete_map=colors,
        title=f"Global Mean Comparison: {metric}",
        height=500,
        template="plotly_white"
    )

    fig_global.update_layout(
        xaxis_title="Stimulus (Image)",
        yaxis_title=f"Mean {metric} (Normalized)",
        legend_title="Data Source",
        font=dict(size=14)
    )

    # Configuração Download
    config_global = {
        'toImageButtonOptions': {
            'format': 'png', 'filename': f'global_comparison_{metric}',
            'height': 600, 'width': 1000, 'scale': 2
        }
    }
    st.plotly_chart(fig_global, use_container_width=True, config=config_global)
    
    with st.expander("View Global Data Table"):
        st.dataframe(df_global.style.format("{:.3f}"))

# ==============================================================================
# ABA 2: ANÁLISE INDIVIDUAL (O teu código original melhorado)
# ==============================================================================
with tab_individual:
    st.header("Individual Athlete Deep-Dive")
    
    # Seletor
    atletas_list = df_final[["ID", "Jogadora"]].drop_duplicates().sort_values("Jogadora")
    nome_sel = st.selectbox("Select Athlete:", atletas_list["Jogadora"])
    id_sel = atletas_list[atletas_list["Jogadora"] == nome_sel]["ID"].values[0]

    df_sel = df_final[df_final["ID"] == id_sel].copy().sort_values("Imagem")
    
    # --- GRÁFICO 1: Barras Integradas ---
    st.subheader(f"📊 Integrated Comparison - {nome_sel}")
    
    # Preparar dados para Facet Plot
    df_val = df_sel[["Imagem", "Valence", "Valence_Inquerito", "Valence_OASIS"]].melt(
        id_vars="Imagem", var_name="Fonte", value_name="Valor")
    df_val["Dimension"] = "Valence"
    
    df_ar = df_sel[["Imagem", "Arousal", "Arousal_Inquerito", "Arousal_OASIS"]].melt(
        id_vars="Imagem", var_name="Fonte", value_name="Valor")
    df_ar["Dimension"] = "Arousal"
    
    df_plot = pd.concat([df_val, df_ar])
    # Limpar nomes das fontes para Inglês
    df_plot["Fonte"] = df_plot["Fonte"].str.replace("Valence_", "").str.replace("Arousal_", "").str.replace("Inquerito", "Survey")
    
    # Ajustar labels das imagens
    df_plot["Image_Label"] = df_plot["Imagem"].apply(lambda x: f"Img {x.split('_')[1]}")

    fig_ind = px.bar(
        df_plot,
        x="Image_Label",
        y="Valor",
        color="Fonte",
        facet_col="Dimension",
        barmode="group",
        height=500,
        title=f"Valence & Arousal by Image - {nome_sel}",
        template="plotly_white",
        color_discrete_map=colors
    )
    fig_ind.update_yaxes(matches=None) # Escalas independentes se necessário
    fig_ind.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    st.plotly_chart(fig_ind, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': f'individual_comparison_{id_sel}', 'scale': 2}})

    # --- GRÁFICOS DE DISPERSÃO (SCATTER) ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("FaceReader vs. OASIS")
        # Valence
        fig_v1 = px.scatter(df_sel, x="Valence_OASIS", y="Valence", text="Imagem", title="Valence")
        fig_v1.add_shape(type="line", x0=-1, x1=1, y0=-1, y1=1, line=dict(dash="dash", color="gray"))
        fig_v1.update_layout(template="plotly_white", xaxis_title="OASIS", yaxis_title="FaceReader")
        st.plotly_chart(fig_v1, use_container_width=True)
        
        # Arousal
        fig_a1 = px.scatter(df_sel, x="Arousal_OASIS", y="Arousal", text="Imagem", title="Arousal")
        fig_a1.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="gray"))
        fig_a1.update_layout(template="plotly_white", xaxis_title="OASIS", yaxis_title="FaceReader")
        st.plotly_chart(fig_a1, use_container_width=True)

    with col2:
        st.subheader("Survey vs. OASIS")
        # Valence
        fig_v2 = px.scatter(df_sel, x="Valence_OASIS", y="Valence_Inquerito", text="Imagem", title="Valence")
        fig_v2.add_shape(type="line", x0=-1, x1=1, y0=-1, y1=1, line=dict(dash="dash", color="gray"))
        fig_v2.update_layout(template="plotly_white", xaxis_title="OASIS", yaxis_title="Survey")
        st.plotly_chart(fig_v2, use_container_width=True)
        
        # Arousal
        fig_a2 = px.scatter(df_sel, x="Arousal_OASIS", y="Arousal_Inquerito", text="Imagem", title="Arousal")
        fig_a2.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="gray"))
        fig_a2.update_layout(template="plotly_white", xaxis_title="OASIS", yaxis_title="Survey")
        st.plotly_chart(fig_a2, use_container_width=True)

    # --- TABELAS ---
    st.subheader("📋 Correlation & Error Summary")
    
    # Cálculos
    corr_val_fr_inq = df_sel[["Valence", "Valence_Inquerito"]].corr().iloc[0, 1]
    corr_ar_fr_inq = df_sel[["Arousal", "Arousal_Inquerito"]].corr().iloc[0, 1]
    
    corr_val_fo = df_sel[["Valence", "Valence_OASIS"]].corr().iloc[0, 1]
    corr_ar_fo = df_sel[["Arousal", "Arousal_OASIS"]].corr().iloc[0, 1]
    
    corr_val_io = df_sel[["Valence_Inquerito", "Valence_OASIS"]].corr().iloc[0, 1]
    corr_ar_io = df_sel[["Arousal_Inquerito", "Arousal_OASIS"]].corr().iloc[0, 1]

    # MAE (Erro Médio Absoluto)
    mae_val = np.mean((df_sel["Valence"] - df_sel["Valence_Inquerito"]).abs())
    mae_ar = np.mean((df_sel["Arousal"] - df_sel["Arousal_Inquerito"]).abs())

    df_stats = pd.DataFrame({
        "Comparison": ["FaceReader vs Survey", "FaceReader vs OASIS", "Survey vs OASIS"],
        "Valence (r)": [corr_val_fr_inq, corr_val_fo, corr_val_io],
        "Arousal (r)": [corr_ar_fr_inq, corr_ar_fo, corr_ar_io],
        "Valence (MAE)": [mae_val, np.mean((df_sel["Valence"] - df_sel["Valence_OASIS"]).abs()), np.mean((df_sel["Valence_Inquerito"] - df_sel["Valence_OASIS"]).abs())],
        "Arousal (MAE)": [mae_ar, np.mean((df_sel["Arousal"] - df_sel["Arousal_OASIS"]).abs()), np.mean((df_sel["Arousal_Inquerito"] - df_sel["Arousal_OASIS"]).abs())]
    })

    st.dataframe(df_stats.style.format("{:.3f}"))

    # Discrepâncias
    df_sel["Val_diff"] = (df_sel["Valence"] - df_sel["Valence_Inquerito"]).abs()
    df_sel["Aro_diff"] = (df_sel["Arousal"] - df_sel["Arousal_Inquerito"]).abs()
    
    # Filtro para grandes erros (> 0.4)
    discrepancias = df_sel[(df_sel["Val_diff"] > 0.4) | (df_sel["Aro_diff"] > 0.4)]

    if not discrepancias.empty:
        st.warning("⚠️ High Discrepancies Detected (FaceReader vs Survey > 0.4)")
        st.dataframe(discrepancias[["Imagem", "Val_diff", "Aro_diff"]].rename(columns={
            "Val_diff": "Diff Valence", "Aro_diff": "Diff Arousal"
        }).style.format("{:.3f}"))


