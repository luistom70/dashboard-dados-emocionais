import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from utils import (
    carregar_dados, carregar_inqueritos, carregar_oasis,
    comparar_face_reader_vs_inquerito, adicionar_distancia_emocional,
    normalizar_oasis
)

# --- FUNÇÃO AUXILIAR ROBUSTA ---
def extrair_numero_imagem(texto):
    """
    Tenta extrair o primeiro número encontrado no nome da imagem.
    Ex: 'Imagem_1' -> 1, 'Img 10' -> 10, '12.jpg' -> 12
    Se não encontrar número, devolve 0 para não dar erro.
    """
    match = re.search(r'\d+', str(texto))
    if match:
        return int(match.group())
    return 0

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

# Renomear colunas do FaceReader para evitar KeyErrors
df_final = df_final.rename(columns={
    "Valence": "Valence_FaceReader",
    "Arousal": "Arousal_FaceReader"
})

# IDs para seleção individual
df["ID"] = df["Atleta"].astype("category").cat.codes + 1
# Garantir que temos o ID no df_final
df_final = df_final.merge(df[["Atleta", "ID"]].drop_duplicates(), left_on="Jogadora", right_on="Atleta")

# --- CRIAÇÃO DAS ABAS ---
tab_global, tab_individual = st.tabs(["🌍 Global Analysis (Paper)", "👤 Individual Analysis"])

# ==============================================================================
# ABA 1: ANÁLISE GLOBAL (Para o Artigo)
# ==============================================================================
with tab_global:
    st.header("Global Trends (All Athletes Aggregated)")
    st.caption("Compare how different aggregation methods affect the alignment between sources.")

    # --- NOVO: SELETOR DE AGREGAÇÃO ---
    col_agg, col_metric = st.columns(2)
    with col_agg:
        agg_method = st.selectbox(
            "Aggregation Method:", 
            ["Mean (Average)", "Max (Peak Positive)", "Min (Peak Negative)"],
            help="Mean: Smooths noise. Max: Good for Arousal/Happiness. Min: Good for Sadness/Fear."
        )
    with col_metric:
        metric = st.radio("Select Metric:", ["Valence", "Arousal"], horizontal=True)

    # Definir função de agregação baseada na escolha
    if "Mean" in agg_method:
        agg_func = 'mean'
    elif "Max" in agg_method:
        agg_func = 'max'
    else: # Min
        agg_func = 'min'

    # Calcular Global usando a função escolhida
    df_global = df_final.groupby("Imagem")[
        ["Valence_FaceReader", "Valence_Inquerito", "Valence_OASIS",
         "Arousal_FaceReader", "Arousal_Inquerito", "Arousal_OASIS"]
    ].agg(agg_func).reset_index()

    # Ordenação Robusta
    df_global["ImgNum"] = df_global["Imagem"].apply(extrair_numero_imagem)
    df_global = df_global.sort_values("ImgNum")
    df_global["Label"] = df_global["ImgNum"].apply(lambda x: f"Img {x}")

    # Preparar dados Plotly
    cols = [f"{metric}_FaceReader", f"{metric}_Inquerito", f"{metric}_OASIS"]
    df_melt = df_global.melt(id_vars=["Label"], value_vars=cols, var_name="Source", value_name="Value")
    
    # Limpeza de nomes
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
        title=f"Global Comparison ({agg_method}): {metric}",
        height=500,
        template="plotly_white"
    )

    fig_global.update_layout(
        xaxis_title="Stimulus (Image)",
        yaxis_title=f"{metric} Value ({agg_method})",
        legend_title="Data Source",
        font=dict(size=14)
    )

    # Configuração Download
    config_global = {
        'toImageButtonOptions': {
            'format': 'png', 'filename': f'global_comparison_{metric}_{agg_method}',
            'height': 600, 'width': 1000, 'scale': 2
        }
    }
    st.plotly_chart(fig_global, use_container_width=True, config=config_global)
    
    # Mostrar Tabela
    with st.expander("View Global Data Table"):
        float_cols = df_global.select_dtypes(include='float').columns
        st.dataframe(df_global.style.format("{:.3f}", subset=float_cols))
        
    # --- DICA DE ANÁLISE ---
    if metric == "Arousal":
        st.info("💡 **Tip for Arousal:** Try selecting **'Max (Peak Positive)'**. Since Arousal is purely intensity (0 to 1), the Maximum usually captures the true reaction better than the Mean.")
    elif metric == "Valence":
        st.info("💡 **Tip for Valence:** \n- For **Happy** images, use **'Max'**.\n- For **Sad/Angry** images, use **'Min'**.\n- The 'Mean' tends to flatten everything towards zero.")

# ==============================================================================
# ABA 2: ANÁLISE INDIVIDUAL
# ==============================================================================
with tab_individual:
    st.header("Individual Athlete Deep-Dive")
    
    atletas_list = df_final[["ID", "Jogadora"]].drop_duplicates().sort_values("Jogadora")
    
    if atletas_list.empty:
        st.error("No athlete data found. Please check your data file.")
        st.stop()

    nome_sel = st.selectbox("Select Athlete:", atletas_list["Jogadora"])
    
    try:
        id_sel = atletas_list[atletas_list["Jogadora"] == nome_sel]["ID"].values[0]
    except IndexError:
        st.error("Error identifying athlete ID.")
        st.stop()

    df_sel = df_final[df_final["ID"] == id_sel].copy()
    
    # Ordenar usando a função robusta
    df_sel["ImgNum"] = df_sel["Imagem"].apply(extrair_numero_imagem)
    df_sel = df_sel.sort_values("ImgNum")
    
    # --- GRÁFICO 1: Barras Integradas ---
    st.subheader(f"📊 Integrated Comparison - {nome_sel}")
    
    colors = {"FaceReader": "#1f77b4", "Survey": "#ff7f0e", "OASIS": "#2ca02c"}

    df_val = df_sel[["Imagem", "Valence_FaceReader", "Valence_Inquerito", "Valence_OASIS"]].melt(
        id_vars="Imagem", var_name="Fonte", value_name="Valor")
    df_val["Dimension"] = "Valence"
    
    df_ar = df_sel[["Imagem", "Arousal_FaceReader", "Arousal_Inquerito", "Arousal_OASIS"]].melt(
        id_vars="Imagem", var_name="Fonte", value_name="Valor")
    df_ar["Dimension"] = "Arousal"
    
    df_plot = pd.concat([df_val, df_ar])
    
    df_plot["Fonte"] = df_plot["Fonte"].str.replace("Valence_", "").str.replace("Arousal_", "").str.replace("Inquerito", "Survey")
    
    df_plot["ImgNum"] = df_plot["Imagem"].apply(extrair_numero_imagem)
    df_plot["Image_Label"] = df_plot["ImgNum"].apply(lambda x: f"Img {x}")
    df_plot = df_plot.sort_values(["Dimension", "ImgNum"])

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
    fig_ind.update_yaxes(matches=None)
    fig_ind.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    st.plotly_chart(fig_ind, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': f'individual_comparison_{id_sel}', 'scale': 2}})

    # --- GRÁFICOS DE DISPERSÃO ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("FaceReader vs. OASIS")
        fig_v1 = px.scatter(df_sel, x="Valence_OASIS", y="Valence_FaceReader", text="Imagem", title="Valence")
        fig_v1.add_shape(type="line", x0=-1, x1=1, y0=-1, y1=1, line=dict(dash="dash", color="gray"))
        fig_v1.update_layout(template="plotly_white", xaxis_title="OASIS", yaxis_title="FaceReader")
        st.plotly_chart(fig_v1, use_container_width=True)
        
        fig_a1 = px.scatter(df_sel, x="Arousal_OASIS", y="Arousal_FaceReader", text="Imagem", title="Arousal")
        fig_a1.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="gray"))
        fig_a1.update_layout(template="plotly_white", xaxis_title="OASIS", yaxis_title="FaceReader")
        st.plotly_chart(fig_a1, use_container_width=True)

    with col2:
        st.subheader("Survey vs. OASIS")
        fig_v2 = px.scatter(df_sel, x="Valence_OASIS", y="Valence_Inquerito", text="Imagem", title="Valence")
        fig_v2.add_shape(type="line", x0=-1, x1=1, y0=-1, y1=1, line=dict(dash="dash", color="gray"))
        fig_v2.update_layout(template="plotly_white", xaxis_title="OASIS", yaxis_title="Survey")
        st.plotly_chart(fig_v2, use_container_width=True)
        
        fig_a2 = px.scatter(df_sel, x="Arousal_OASIS", y="Arousal_Inquerito", text="Imagem", title="Arousal")
        fig_a2.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="gray"))
        fig_a2.update_layout(template="plotly_white", xaxis_title="OASIS", yaxis_title="Survey")
        st.plotly_chart(fig_a2, use_container_width=True)

    # --- TABELAS ---
    st.subheader("📋 Correlation & Error Summary")
    
    def safe_corr(df, col1, col2):
        if df.empty: return 0.0
        val = df[[col1, col2]].corr().iloc[0, 1]
        return val if not np.isnan(val) else 0.0

    corr_val_fr_inq = safe_corr(df_sel, "Valence_FaceReader", "Valence_Inquerito")
    corr_ar_fr_inq = safe_corr(df_sel, "Arousal_FaceReader", "Arousal_Inquerito")
    
    corr_val_fo = safe_corr(df_sel, "Valence_FaceReader", "Valence_OASIS")
    corr_ar_fo = safe_corr(df_sel, "Arousal_FaceReader", "Arousal_OASIS")
    
    corr_val_io = safe_corr(df_sel, "Valence_Inquerito", "Valence_OASIS")
    corr_ar_io = safe_corr(df_sel, "Arousal_Inquerito", "Arousal_OASIS")

    mae_val = np.mean((df_sel["Valence_FaceReader"] - df_sel["Valence_Inquerito"]).abs())
    mae_ar = np.mean((df_sel["Arousal_FaceReader"] - df_sel["Arousal_Inquerito"]).abs())

    df_stats = pd.DataFrame({
        "Comparison": ["FaceReader vs Survey", "FaceReader vs OASIS", "Survey vs OASIS"],
        "Valence (r)": [corr_val_fr_inq, corr_val_fo, corr_val_io],
        "Arousal (r)": [corr_ar_fr_inq, corr_ar_fo, corr_ar_io],
        "Valence (MAE)": [
            mae_val, 
            np.mean((df_sel["Valence_FaceReader"] - df_sel["Valence_OASIS"]).abs()), 
            np.mean((df_sel["Valence_Inquerito"] - df_sel["Valence_OASIS"]).abs())
        ],
        "Arousal (MAE)": [
            mae_ar, 
            np.mean((df_sel["Arousal_FaceReader"] - df_sel["Arousal_OASIS"]).abs()), 
            np.mean((df_sel["Arousal_Inquerito"] - df_sel["Arousal_OASIS"]).abs())
        ]
    })

    # --- CORREÇÃO DO ERRO AQUI ---
    # Só formatamos as colunas numéricas
    float_cols_stats = df_stats.select_dtypes(include='float').columns
    st.dataframe(df_stats.style.format("{:.3f}", subset=float_cols_stats))

    # Discrepâncias
    df_sel["Val_diff"] = (df_sel["Valence_FaceReader"] - df_sel["Valence_Inquerito"]).abs()
    df_sel["Aro_diff"] = (df_sel["Arousal_FaceReader"] - df_sel["Arousal_Inquerito"]).abs()
    
    discrepancias = df_sel[(df_sel["Val_diff"] > 0.4) | (df_sel["Aro_diff"] > 0.4)]

    if not discrepancias.empty:
        st.warning("⚠️ High Discrepancies Detected (FaceReader vs Survey > 0.4)")
        # Selecionar apenas as colunas que queremos mostrar e renomear
        df_discr_show = discrepancias[["Imagem", "Val_diff", "Aro_diff"]].rename(columns={
            "Val_diff": "Diff Valence", "Aro_diff": "Diff Arousal"
        })
        # --- CORREÇÃO DO ERRO AQUI ---
        # Aplicamos formatação apenas nas colunas numéricas (Diff Valence e Diff Arousal)
        st.dataframe(df_discr_show.style.format("{:.3f}", subset=["Diff Valence", "Diff Arousal"]))


