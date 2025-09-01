import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from utils import (
    carregar_dados, carregar_inqueritos,
    comparar_face_reader_vs_inquerito, adicionar_distancia_emocional
)

st.set_page_config(layout="wide")
st.title("🔮 Modelo Preditivo de FaceReader a partir do Inquérito")

arquivo = st.session_state.get("arquivo")
if not arquivo:
    st.warning("Por favor, carrega o ficheiro na página Home.")
    st.stop()

# --- Carregamento e pré-processamento
df = carregar_dados(arquivo)
df_inq = carregar_inqueritos(arquivo)
df_comparado = comparar_face_reader_vs_inquerito(df, df_inq)
df_comparado = adicionar_distancia_emocional(df_comparado)

df_comparado["Imagem"] = df_comparado["Imagem"].astype(str).str.strip()

# --- Interface de seleção
st.subheader("⚙️ Parâmetros do Modelo")

alvo = st.selectbox("Variável a prever (FaceReader):", ["Valence", "Arousal"])
modelo_nome = st.selectbox(
    "Seleciona o algoritmo:",
    ["Linear", "Ridge", "Random Forest", "Gradient Boosting", "XGBoost", "SVR", "Rede Neural (MLP)"]
)

st.markdown("📌 Este modelo usa apenas os valores do **Inquérito** como entrada.")

# --- Definir X (features) e y (target)
X = df_comparado[["Valence_Inquerito", "Arousal_Inquerito"]]
y = df_comparado[alvo]

# --- Treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Escolher modelo
if modelo_nome == "Linear":
    modelo = LinearRegression()
elif modelo_nome == "Ridge":
    modelo = Ridge(alpha=1.0)
elif modelo_nome == "Random Forest":
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
elif modelo_nome == "Gradient Boosting":
    modelo = GradientBoostingRegressor(n_estimators=100, random_state=42)
elif modelo_nome == "XGBoost":
    modelo = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
elif modelo_nome == "Rede Neural (MLP)":
    modelo = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
else:
    modelo = SVR()

# --- Treinar e prever
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# --- Avaliação
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Avaliação do Modelo")
st.markdown(f"""
- **Modelo**: `{modelo_nome}`  
- **Erro Médio Absoluto (MAE)**: `{mae:.3f}`  
- **Coeficiente de Determinação (R²)**: `{r2:.3f}`
""")

# --- Gráfico Real vs Previsto
st.subheader("📈 Comparação: Real vs Previsto")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, y_pred, alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
ax.set_title(f"{alvo} - {modelo_nome}: Real vs Previsto")
ax.set_xlabel("Valor Real")
ax.set_ylabel("Valor Previsto")
ax.grid(True)
st.pyplot(fig)


