import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import (
    carregar_dados, carregar_inqueritos, carregar_oasis,
    comparar_face_reader_vs_inquerito, adicionar_distancia_emocional,
    normalizar_oasis, calcular_distancias_ao_oasis
)

st.set_page_config(layout="wide")
st.title("📊 Comparações FaceReader, Inquérito e OASIS")

# Função para gráfico comparativo com facet

def plot_comparacao_por_imagem_valence_arousal(df_comparado, jogadora_id):
    df_sel = df_comparado[df_comparado["ID"] == jogadora_id].copy()
    df_sel = df_sel.sort_values("Imagem")

    df_val = df_sel[["Imagem", "Valence", "Valence_Inquerito", "Valence_OASIS"]].melt(
        id_vars="Imagem", var_name="Fonte", value_name="Valor")
    df_val["Dimensão"] = "Valence"
    df_val["Fonte"] = df_val["Fonte"].str.replace("Valence_", "", regex=False)

    df_ar = df_sel[["Imagem", "Arousal", "Arousal_Inquerito", "Arousal_OASIS"]].melt(
        id_vars="Imagem", var_name="Fonte", value_name="Valor")
    df_ar["Dimensão"] = "Arousal"
    df_ar["Fonte"] = df_ar["Fonte"].str.replace("Arousal_", "", regex=False)

    df_plot = pd.concat([df_val, df_ar])

    fig = px.bar(
        df_plot,
        x="Imagem",
        y="Valor",
        color="Fonte",
        facet_col="Dimensão",
        barmode="group",
        height=500,
        title=f"Comparação de Valence e Arousal por Imagem - ID {jogadora_id}"
    )
    fig.update_layout(
        xaxis=dict(linecolor="black"),
        xaxis2=dict(linecolor="black"),
        yaxis=dict(linecolor="black"),
        yaxis2=dict(linecolor="black")
    )
    st.plotly_chart(fig, use_container_width=True)

# Interface principal

arquivo = st.session_state.get("arquivo")
if not arquivo:
    st.warning("Por favor, carrega o ficheiro na página Home.")
    st.stop()

if arquivo:
    df = carregar_dados(arquivo)
    df_inq = carregar_inqueritos(arquivo)
    df_oasis = carregar_oasis(arquivo)
    df_oasis = normalizar_oasis(df_oasis)

    df_comparado = comparar_face_reader_vs_inquerito(df, df_inq)
    df_comparado = adicionar_distancia_emocional(df_comparado)

    df_comparado["Imagem"] = df_comparado["Imagem"].astype(str).str.strip()
    df_oasis["Imagem"] = df_oasis["Imagem"].astype(str).str.strip()
    df_comparado = df_comparado.merge(df_oasis, on="Imagem", how="left")
    df["ID"] = df["Atleta"].astype("category").cat.codes + 1
    df_comparado = df_comparado.merge(df[["Atleta", "ID"]].drop_duplicates(), left_on="Jogadora", right_on="Atleta")

    ids = sorted(df_comparado["ID"].unique())
    id_sel = st.selectbox("Seleciona uma atleta (ID):", ids)

    df_sel = df_comparado[df_comparado["ID"] == id_sel].copy()

    st.subheader("📊 Comparação Integrada Valence e Arousal")
    plot_comparacao_por_imagem_valence_arousal(df_comparado, id_sel)

    # Gráficos de dispersão com eixos a preto
    st.subheader(f"📌 Dispersão FaceReader vs OASIS - ID {id_sel}")
    corr_val_fo = df_comparado[df_comparado["ID"] == id_sel][["Valence", "Valence_OASIS"]].corr().iloc[0, 1]
    corr_ar_fo = df_comparado[df_comparado["ID"] == id_sel][["Arousal", "Arousal_OASIS"]].corr().iloc[0, 1]

    fig_val_fo = px.scatter(df_comparado[df_comparado["ID"] == id_sel],
                            x="Valence_OASIS", y="Valence", text="Imagem",
                            title=f"Valence: OASIS vs FaceReader (r = {corr_val_fo:.2f})")
    fig_val_fo.update_traces(marker=dict(size=10, color="blue"), textposition="top center", textfont=dict(color="black"))
    fig_val_fo.add_shape(type="line", x0=-1, x1=1, y0=-1, y1=1, line=dict(dash="dash", color="gray"))
    fig_val_fo.update_layout(
        xaxis=dict(linecolor="black", range=[-1, 1]),
        yaxis=dict(linecolor="black", range=[-1, 1])
    )
    st.plotly_chart(fig_val_fo, use_container_width=True)

    fig_ar_fo = px.scatter(df_comparado[df_comparado["ID"] == id_sel],
                           x="Arousal_OASIS", y="Arousal", text="Imagem",
                           title=f"Arousal: OASIS vs FaceReader (r = {corr_ar_fo:.2f})")
    fig_ar_fo.update_traces(marker=dict(size=10, color="green"), textposition="top center", textfont=dict(color="black"))
    fig_ar_fo.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="gray"))
    fig_ar_fo.update_layout(
        xaxis=dict(linecolor="black", range=[0, 1]),
        yaxis=dict(linecolor="black", range=[0, 1])
    )
    st.plotly_chart(fig_ar_fo, use_container_width=True)

        # Dispersão Inquérito vs OASIS
    st.subheader(f"📌 Dispersão Inquérito vs OASIS - ID {id_sel}")
    corr_val_io = df_comparado[df_comparado["ID"] == id_sel][["Valence_Inquerito", "Valence_OASIS"]].corr().iloc[0, 1]
    corr_ar_io = df_comparado[df_comparado["ID"] == id_sel][["Arousal_Inquerito", "Arousal_OASIS"]].corr().iloc[0, 1]

    fig_val_io = px.scatter(df_comparado[df_comparado["ID"] == id_sel],
                            x="Valence_OASIS", y="Valence_Inquerito", text="Imagem",
                            title=f"Valence: OASIS vs Inquérito (r = {corr_val_io:.2f})")
    fig_val_io.update_traces(marker=dict(size=10, color="purple"), textposition="top center", textfont=dict(color="black"))
    fig_val_io.update_layout(
        xaxis=dict(linecolor="black", range=[-1, 1]),
        yaxis=dict(linecolor="black", range=[-1, 1])
    )
    st.plotly_chart(fig_val_io, use_container_width=True)

    fig_ar_io = px.scatter(df_comparado[df_comparado["ID"] == id_sel],
                           x="Arousal_OASIS", y="Arousal_Inquerito", text="Imagem",
                           title=f"Arousal: OASIS vs Inquérito (r = {corr_ar_io:.2f})")
    fig_ar_io.update_traces(marker=dict(size=10, color="darkorange"), textposition="top center", textfont=dict(color="black"))
    fig_ar_io.update_layout(
        xaxis=dict(linecolor="black", range=[0, 1]),
        yaxis=dict(linecolor="black", range=[0, 1])
    )
    st.plotly_chart(fig_ar_io, use_container_width=True)

    # (restante código permanece inalterado)




    # Tabela resumo de correlações e erros
    st.subheader("📋 Resumo de Correlações e Erros")
    corr_val_fr_inq = df_sel[["Valence", "Valence_Inquerito"]].corr().iloc[0, 1]
    corr_ar_fr_inq = df_sel[["Arousal", "Arousal_Inquerito"]].corr().iloc[0, 1]
    mae_val = np.mean((df_sel["Valence"] - df_sel["Valence_Inquerito"]).abs())
    mae_ar = np.mean((df_sel["Arousal"] - df_sel["Arousal_Inquerito"]).abs())

    corr_val_fr_oas = df_sel[["Valence", "Valence_OASIS"]].corr().iloc[0, 1]
    corr_ar_fr_oas = df_sel[["Arousal", "Arousal_OASIS"]].corr().iloc[0, 1]

    df_corr = pd.DataFrame({
    "Comparação": ["FaceReader vs Inquérito", "FaceReader vs OASIS", "Inquérito vs OASIS"],
    "Valence (r)": [corr_val_fr_inq, corr_val_fr_oas, corr_val_io],
    "Arousal (r)": [corr_ar_fr_inq, corr_ar_fr_oas, corr_ar_io],
    "Valence (MAE)": [
        mae_val,
        np.mean((df_sel["Valence"] - df_sel["Valence_OASIS"]).abs()),
        np.mean((df_sel["Valence_Inquerito"] - df_sel["Valence_OASIS"]).abs())
    ],
    "Arousal (MAE)": [
        mae_ar,
        np.mean((df_sel["Arousal"] - df_sel["Arousal_OASIS"]).abs()),
        np.mean((df_sel["Arousal_Inquerito"] - df_sel["Arousal_OASIS"]).abs())
    ]
    })
    
    st.dataframe(df_corr.style.format({
    "Valence (r)": "{:.3f}",
    "Arousal (r)": "{:.3f}",
    "Valence (MAE)": "{:.3f}",
    "Arousal (MAE)": "{:.3f}"
    }))

    # Destacar discrepâncias maiores que 0.3
    df_sel["Val_diff_inq"] = (df_sel["Valence"] - df_sel["Valence_Inquerito"]).abs()
    df_sel["Aro_diff_inq"] = (df_sel["Arousal"] - df_sel["Arousal_Inquerito"]).abs()
    discrepancias = df_sel[(df_sel["Val_diff_inq"] > 0.3) | (df_sel["Aro_diff_inq"] > 0.3)]

    if not discrepancias.empty:
        st.subheader("⚠️ Discrepâncias Elevadas (FaceReader vs Inquérito)")
        st.dataframe(discrepancias[["Imagem", "Val_diff_inq", "Aro_diff_inq"]].rename(columns={
    "Val_diff_inq": "Diferença Valence",
    "Aro_diff_inq": "Diferença Arousal"
}).style.format({
    "Diferença Valence": "{:.3f}",
    "Diferença Arousal": "{:.3f}"
}))




