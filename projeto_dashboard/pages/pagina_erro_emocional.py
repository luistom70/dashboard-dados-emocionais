import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

from utils import (
    carregar_dados, carregar_inqueritos,
    comparar_face_reader_vs_inquerito, adicionar_distancia_emocional,
    carregar_oasis, normalizar_oasis
)

# --- SETUP
st.set_page_config(layout="wide")
st.title("🔍 Classificação de Erro Emocional: Grande vs Pequeno")

arquivo = st.session_state.get("arquivo")
if not arquivo:
    st.warning("Carrega o ficheiro na página Home.")
    st.stop()

# --- Dados base
df = carregar_dados(arquivo)
df_inq = carregar_inqueritos(arquivo)
df_oasis = carregar_oasis(arquivo)
df_oasis = normalizar_oasis(df_oasis)
df_comparado = comparar_face_reader_vs_inquerito(df, df_inq)
df_comparado = adicionar_distancia_emocional(df_comparado)
df_comparado["Imagem"] = df_comparado["Imagem"].astype(str).str.strip()
df_oasis["Imagem"] = df_oasis["Imagem"].astype(str).str.strip()

# --- Parâmetros
st.subheader("⚙️ Configuração")
target = st.selectbox("Erro a classificar:", ["Valence", "Arousal"])
limiar = st.slider("Limiar de erro considerado 'alto'", 0.1, 1.0, 0.3, step=0.05)

# --- Preparar dados
df_comparado["Erro"] = abs(df_comparado[target] - df_comparado[f"{target}_Inquerito"])
df_comparado["Erro_Alvo"] = (df_comparado["Erro"] > limiar).astype(int)
X = df_comparado[["Valence_Inquerito", "Arousal_Inquerito"]]
y = df_comparado["Erro_Alvo"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Modelo fixo: XGBoost
modelo = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
probas = modelo.predict_proba(X_test)[:, 1]

# --- Avaliação
st.subheader("📊 Avaliação do Modelo")
acc = accuracy_score(y_test, y_pred)
st.markdown(f"**Acurácia:** `{acc:.2%}`")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pequeno", "Grande"], yticklabels=["Pequeno", "Grande"])
ax.set_xlabel("Predito")
ax.set_ylabel("Real")
ax.set_title("Matriz de Confusão")
st.pyplot(fig_cm)
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

# --- Reconstruir dataframe de previsão
df_resultado = X_test.copy()
df_resultado["Erro_Real"] = y_test.values
df_resultado["Erro_Predito"] = y_pred
df_resultado["Prob_Erro_Alto"] = probas
df_resultado = df_resultado.merge(
    df_comparado[["Jogadora", "Imagem", "Erro"]],
    left_index=True,
    right_index=True,
    how="left"
)

# --- Curva ROC
fpr, tpr, _ = roc_curve(y_test, probas)
auc = roc_auc_score(y_test, probas)
st.subheader("📈 Curva ROC")
fig_roc, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("Falsos Positivos")
ax.set_ylabel("Verdadeiros Positivos")
ax.legend()
ax.set_title("Curva ROC - Erro Grande")
st.pyplot(fig_roc)

# --- Tabela de casos com erro alto previsto
st.subheader("📋 Casos com Erro Alto Previsto")
df_altos = df_resultado[df_resultado["Erro_Predito"] == 1]
st.dataframe(df_altos[["Jogadora", "Imagem", "Erro", "Erro_Real", "Prob_Erro_Alto"]].sort_values("Prob_Erro_Alto", ascending=False))
csv_data = df_altos.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv_data, "erros_altos_previstos.csv", "text/csv")

# --- Análise por jogadora e imagem
df_resultado["Erro_Dif"] = abs(df_resultado["Erro_Real"] - df_resultado["Erro_Predito"])

st.subheader("🧠 Erro Médio por Jogadora")
erro_jog = df_resultado.groupby("Jogadora")["Erro_Dif"].mean().reset_index().sort_values("Erro_Dif", ascending=False)
st.dataframe(erro_jog.style.format({"Erro_Dif": "{:.2f}"}))
st.plotly_chart(px.bar(erro_jog, x="Jogadora", y="Erro_Dif", title="Erro Médio Absoluto por Jogadora"), use_container_width=True)

st.subheader("🖼️ Erro Médio por Imagem")
erro_img = df_resultado.groupby("Imagem")["Erro_Dif"].mean().reset_index().sort_values("Erro_Dif", ascending=False)
st.dataframe(erro_img.style.format({"Erro_Dif": "{:.2f}"}))
st.plotly_chart(px.bar(erro_img, x="Imagem", y="Erro_Dif", title="Erro Médio Absoluto por Imagem"), use_container_width=True)

# --- Juntar com OASIS
df_resultado = df_resultado.merge(
    df_oasis[["Imagem", "Valence_OASIS", "Arousal_OASIS"]],
    on="Imagem",
    how="left"
)

def definir_quadrante(val, aro):
    if val >= 0 and aro >= 0.5:
        return "Q1 (positivo-alto)"
    elif val < 0 and aro >= 0.5:
        return "Q2 (negativo-alto)"
    elif val < 0 and aro < 0.5:
        return "Q3 (negativo-baixo)"
    else:
        return "Q4 (positivo-baixo)"

df_resultado["Quadrante_OASIS"] = df_resultado.apply(lambda row: definir_quadrante(row["Valence_OASIS"], row["Arousal_OASIS"]), axis=1)

st.subheader("📊 Erro Médio por Quadrante OASIS")
erro_quad = df_resultado.groupby("Quadrante_OASIS")["Erro_Dif"].mean().reset_index()
st.dataframe(erro_quad.style.format({"Erro_Dif": "{:.2f}"}))
st.plotly_chart(px.bar(erro_quad, x="Quadrante_OASIS", y="Erro_Dif", title="Erro Médio por Quadrante OASIS"), use_container_width=True)

# -------------------------------
# 🔥 Heatmap: Erro por Jogadora e Quadrante OASIS
# -------------------------------
st.subheader("🔥 Heatmap: Erro Médio por Jogadora e Quadrante OASIS")

# Agrupar erro médio por Jogadora x Quadrante OASIS
pivot_erro = df_resultado.pivot_table(
    values="Erro_Dif",
    index="Jogadora",
    columns="Quadrante_OASIS",
    aggfunc="mean"
).fillna(0)

# Ordenar por erro total (soma das colunas)
pivot_erro["Erro_Total"] = pivot_erro.sum(axis=1)
pivot_erro = pivot_erro.sort_values("Erro_Total", ascending=False).drop(columns="Erro_Total")

# Plot com seaborn
fig_heat, ax = plt.subplots(figsize=(10, max(4, len(pivot_erro)*0.5)))
sns.heatmap(pivot_erro, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": "Erro Médio"})
ax.set_title("Erro Médio por Jogadora e Quadrante OASIS")
st.pyplot(fig_heat)
