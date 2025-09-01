import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

from utils import (
    carregar_dados, carregar_inqueritos,
    comparar_face_reader_vs_inquerito, adicionar_distancia_emocional
)

def definir_quadrante(valence, arousal):
    if valence >= 0 and arousal >= 0.5:
        return "Q1"
    elif valence < 0 and arousal >= 0.5:
        return "Q2"
    elif valence < 0 and arousal < 0.5:
        return "Q3"
    else:
        return "Q4"

st.set_page_config(layout="wide")
st.title("🛍️ Modelo de Concordância de Quadrantes")

arquivo = st.session_state.get("arquivo")
if not arquivo:
    st.warning("Por favor, carrega o ficheiro na página Home.")
    st.stop()

# --- Dados
df = carregar_dados(arquivo)
df_inq = carregar_inqueritos(arquivo)
df_comparado = comparar_face_reader_vs_inquerito(df, df_inq)
df_comparado = adicionar_distancia_emocional(df_comparado)

df_comparado["Quadrante_FR"] = df_comparado.apply(lambda row: definir_quadrante(row["Valence"], row["Arousal"]), axis=1)
df_comparado["Quadrante_INQ"] = df_comparado.apply(lambda row: definir_quadrante(row["Valence_Inquerito"], row["Arousal_Inquerito"]), axis=1)
df_comparado["Concorda"] = (df_comparado["Quadrante_FR"] == df_comparado["Quadrante_INQ"]).astype(int)

# --- Features e target
X = df_comparado[["Valence_Inquerito", "Arousal_Inquerito"]]
y = df_comparado["Concorda"]

# --- Treino com SVC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = SVC(kernel='rbf', probability=True, class_weight="balanced")
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# --- Avaliação
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("📊 Avaliação do Modelo")
st.markdown(f"""
- **Modelo:** `SVC (RBF)`  
- **Acurácia:** `{acc:.2%}`
""")

# --- Matriz de confusão
st.subheader("📉 Matriz de Confusão")
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Não", "Sim"], yticklabels=["Não", "Sim"])
ax.set_xlabel("Predito")
ax.set_ylabel("Real")
ax.set_title("Concordância de Quadrantes")
st.pyplot(fig_cm)

# --- Relatório de classificação
st.subheader("📋 Relatório de Classificação")
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report.style.format("{:.2f}"))

# --- Análise de erros
df_resultado = X_test.copy()
df_resultado["Concorda_Real"] = y_test.values
df_resultado["Concorda_Predita"] = y_pred
df_resultado["Correto"] = df_resultado["Concorda_Real"] == df_resultado["Concorda_Predita"]

df_resultado = df_resultado.merge(
    df_comparado[["Jogadora", "Imagem"]],
    left_index=True,
    right_index=True,
    how="left"
)

df_erros = df_resultado[df_resultado["Correto"] == False]
st.subheader("🔍 Casos em que o modelo errou")
st.write(f"Total de erros: {len(df_erros)}")
st.dataframe(df_erros[["Jogadora", "Imagem", "Concorda_Real", "Concorda_Predita"]])

st.subheader("📊 Frequência de Erros por Jogadora")
erros_por_jogadora = df_erros["Jogadora"].value_counts().reset_index()
erros_por_jogadora.columns = ["Jogadora", "Erros"]
st.dataframe(erros_por_jogadora)

st.subheader("🖼️ Frequência de Erros por Imagem")
erros_por_imagem = df_erros["Imagem"].value_counts().reset_index()
erros_por_imagem.columns = ["Imagem", "Erros"]
st.dataframe(erros_por_imagem)


