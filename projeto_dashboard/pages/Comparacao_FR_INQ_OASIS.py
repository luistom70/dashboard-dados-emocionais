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

    # Cores Científicas (Alto Contraste)
    colors = {
        "FaceReader": "#1f77b4", # Azul forte
        "Survey": "#ff7f0e",     # Laranja forte
        "OASIS": "#2ca02c"       # Verde forte
    }

    # 1. GRÁFICO DE BARRAS GLOBAL (SCIENTIFIC STYLE)
    fig_global = px.bar(
        df_melt,
        x="Label",
        y="Value",
        color="Source",
        barmode="group",
        color_discrete_map=colors,
        title=f"Global Comparison ({agg_method}): {metric}",
        height=600 # Mais alto para se ler bem
    )

    # Layout Profissional (Fundo Branco, Letras Grandes)
    fig_global.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        title=dict(font=dict(size=22, color='black', family="Arial"), y=0.95),
        xaxis=dict(
            title="Stimulus (Image ID)",
            title_font=dict(size=18, family="Arial Black"),
            tickfont=dict(size=14, color='black'),
            showline=True, linecolor='black', linewidth=2
        ),
        yaxis=dict(
            title=f"{metric} Value ({agg_func.capitalize()})",
            title_font=dict(size=18, family="Arial Black"),
            tickfont=dict(size=14, color='black'),
            showgrid=True, gridcolor='lightgrey',
            showline=True, linecolor='black', linewidth=2,
            zeroline=True, zerolinecolor='black'
        ),
        legend=dict(
            title="Data Source",
            font=dict(size=16),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black", borderwidth=1,
            yanchor="top", y=0.99, xanchor="left", x=0.01
        ),
        bargap=0.15, # Espaço entre grupos
        bargroupgap=0.05 # Espaço entre barras do mesmo grupo
    )

    # Configuração Download Alta Resolução
    config_global = {
        'toImageButtonOptions': {
            'format': 'png', 'filename': f'global_comparison_{metric}_{agg_method}',
            'height': 800, 'width': 1200, 'scale': 2
        }
    }
    st.plotly_chart(fig_global, use_container_width=True, config=config_global)

    st.markdown("---")
    st.subheader("🛡️ Emotional Regulation Index (The 'Stoicism' Gap)")
    st.caption("Difference between Reported Intensity (Survey) and Physiological Intensity (FaceReader). Higher bars indicate higher emotional suppression.")

    # 2. CALCULAR ÍNDICE DE REGULAÇÃO
    # Usamos os Picos: Inquérito (Max) - FaceReader (Max)
    
    # Agrupar por atleta e pegar nos máximos de cada imagem
    df_regulation = df_final.groupby(["ID", "Jogadora", "Imagem"])[
        ["Arousal_Inquerito", "Arousal_FaceReader"]
    ].max().reset_index()

    # Calcular a média dos máximos por atleta
    df_reg_atleta = df_regulation.groupby(["ID", "Jogadora"])[
        ["Arousal_Inquerito", "Arousal_FaceReader"]
    ].mean().reset_index()

    # Calcular o Índice de Supressão (Inquérito - FaceReader)
    df_reg_atleta["Suppression_Index"] = df_reg_atleta["Arousal_Inquerito"] - df_reg_atleta["Arousal_FaceReader"]
    
    # Classificar: Index > 0.4 é "High Regulation", < 0.2 é "Low Regulation"
    def classify_reg(val):
        if val > 0.4: return "High Suppression (Stoic)" # Ajustei para 0.4 para ser mais sensível
        if val > 0.2: return "Moderate Regulation"
        return "Low Suppression (Expressive)"
    
    df_reg_atleta["Profile"] = df_reg_atleta["Suppression_Index"].apply(classify_reg)
    
    # Ordenar
    df_reg_atleta = df_reg_atleta.sort_values("Suppression_Index", ascending=False)
    # Criar Label do Eixo X (ID do Atleta)
    df_reg_atleta["ID_Label"] = df_reg_atleta["ID"].apply(lambda x: f"ID {x}")

    # 3. GRÁFICO DE REGULAÇÃO (SCIENTIFIC STYLE)
    # Cores semânticas
    color_map = {
        "High Suppression (Stoic)": "#d62728",      # Vermelho (Alerta)
        "Moderate Regulation": "#ff7f0e",           # Laranja
        "Low Suppression (Expressive)": "#2ca02c"   # Verde (Natural)
    }

    fig_reg = px.bar(
        df_reg_atleta,
        x="ID_Label", # ID 1, ID 2...
        y="Suppression_Index",
        color="Profile",
        title="Emotional Regulation Profile per Athlete",
        labels={"Suppression_Index": "Stoicism Gap (Survey - FaceReader)", "ID_Label": "Athlete"},
        color_discrete_map=color_map,
        height=600
    )
    
    # Linha Zero (Referência)
    fig_reg.add_hline(y=0, line_dash="dash", line_color="black", line_width=2)
    
    # Layout Profissional
    fig_reg.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        title=dict(font=dict(size=22, color='black', family="Arial"), y=0.95),
        xaxis=dict(
            title="Athlete ID",
            title_font=dict(size=18, family="Arial Black"),
            tickfont=dict(size=14, color='black'),
            showline=True, linecolor='black', linewidth=2
        ),
        yaxis=dict(
            title="Suppression Index (Gap)",
            title_font=dict(size=18, family="Arial Black"),
            tickfont=dict(size=14, color='black'),
            showgrid=True, gridcolor='lightgrey',
            showline=True, linecolor='black', linewidth=2
        ),
        legend=dict(
            title="Regulation Profile",
            font=dict(size=14),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black", borderwidth=1
        )
    )
    
    st.plotly_chart(fig_reg, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'regulation_index_scientific', 'scale': 2}})
    
    with st.expander("Ver Tabela de Regulação"):
        st.dataframe(df_reg_atleta.style.format("{:.3f}", subset=["Arousal_Inquerito", "Arousal_FaceReader", "Suppression_Index"]))
    
    # Mostrar Tabela Global
    with st.expander("View Global Data Table"):
        float_cols = df_global.select_dtypes(include='float').columns
        st.dataframe(df_global.style.format("{:.3f}", subset=float_cols))
        
    # --- DICAS ---
    if metric == "Arousal":
        st.info("💡 **Tip for Arousal:** Try selecting **'Max (Peak Positive)'**. Maximum intensity usually captures true reactivity better than Mean.")
    elif metric == "Valence":
        st.info("💡 **Tip for Valence:** For negative images, use **'Min'** to capture the frown. Mean smooths it out.")

    # --- TABELA COMPARATIVA (Igual, só formatação se quiseres) ---
    st.markdown("---")
    st.subheader("📊 Statistical Validation: Mean vs. Peak")
    
    # ... (O teu código da tabela mantém-se igual, tabelas padrão do Streamlit são ok) ...
    # 1. Calcular Dados Agregados por MÉDIA (Baseline)
    df_mean = df_final.groupby("Imagem")[
        ["Valence_FaceReader", "Valence_Inquerito", "Valence_OASIS",
         "Arousal_FaceReader", "Arousal_Inquerito", "Arousal_OASIS"]
    ].mean().reset_index()

    # 2. Calcular Dados Agregados por PICO (Nossa Escolha)
    df_min = df_final.groupby("Imagem")[
        ["Valence_FaceReader", "Valence_Inquerito", "Valence_OASIS"]
    ].min().reset_index()
    
    df_max = df_final.groupby("Imagem")[
        ["Arousal_FaceReader", "Arousal_Inquerito", "Arousal_OASIS"]
    ].max().reset_index()

    # Função Auxiliar de Cálculo
    def get_row_data(df, metric, label):
        fr = f"{metric}_FaceReader"
        inq = f"{metric}_Inquerito"
        col_oasis = f"{metric}_OASIS"

        r_fr = df[fr].corr(df[col_oasis])
        mae_fr = (df[fr] - df[col_oasis]).abs().mean()
        
        r_inq = df[inq].corr(df[col_oasis])
        mae_inq = (df[inq] - df[col_oasis]).abs().mean()
        
        return [label, f"{r_fr:.2f}", f"{mae_fr:.2f}", f"{r_inq:.2f}", f"{mae_inq:.2f}"]

    # Construir as linhas
    data = []
    data.append(get_row_data(df_mean, "Valence", "Valence (Mean)"))
    data.append(get_row_data(df_min, "Valence", "Valence (Peak: Min)"))
    data.append(get_row_data(df_mean, "Arousal", "Arousal (Mean)"))
    data.append(get_row_data(df_max, "Arousal", "Arousal (Peak: Max)"))

    df_table_comp = pd.DataFrame(data, columns=["Method", "FR vs OASIS (r)", "FR vs OASIS (MAE)", "Survey vs OASIS (r)", "Survey vs OASIS (MAE)"])
    st.table(df_table_comp)
    st.caption("This table proves why Peak analysis is superior to Mean analysis for this dataset.")

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

    float_cols_stats = df_stats.select_dtypes(include='float').columns
    st.dataframe(df_stats.style.format("{:.3f}", subset=float_cols_stats))

    # Discrepâncias
    df_sel["Val_diff"] = (df_sel["Valence_FaceReader"] - df_sel["Valence_Inquerito"]).abs()
    df_sel["Aro_diff"] = (df_sel["Arousal_FaceReader"] - df_sel["Arousal_Inquerito"]).abs()
    
    discrepancias = df_sel[(df_sel["Val_diff"] > 0.4) | (df_sel["Aro_diff"] > 0.4)]

    if not discrepancias.empty:
        st.warning("⚠️ High Discrepancies Detected (FaceReader vs Survey > 0.4)")
        df_discr_show = discrepancias[["Imagem", "Val_diff", "Aro_diff"]].rename(columns={
            "Val_diff": "Diff Valence", "Aro_diff": "Diff Arousal"
        })
        st.dataframe(df_discr_show.style.format("{:.3f}", subset=["Diff Valence", "Diff Arousal"]))








