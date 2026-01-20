import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap  # Necessário instalar: pip install shap

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor

# Importar as tuas funções (mantive igual)
from utils import (
    carregar_dados, carregar_inqueritos,
    comparar_face_reader_vs_inquerito, adicionar_distancia_emocional
)

st.set_page_config(layout="wide")
st.title("🔮 Modelo Preditivo (Com Validação Científica)")

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
st.sidebar.header("⚙️ Configuração")
alvo = st.sidebar.selectbox("Variável a prever (Target - FaceReader):", ["Valence", "Arousal"])
modelo_nome = st.sidebar.selectbox(
    "Algoritmo:",
    ["XGBoost", "Random Forest", "Gradient Boosting", "Linear", "Ridge", "SVR", "Rede Neural (MLP)"]
)

st.markdown(f"### 🎯 A tentar prever: **{alvo} (FaceReader)** usando apenas dados subjetivos.")

# --- Definir X (features) e y (target)
X = df_comparado[["Valence_Inquerito", "Arousal_Inquerito"]]
y = df_comparado[alvo]

# --- 1. VALIDAÇÃO CIENTÍFICA (CROSS-VALIDATION) ---
# Isto é o que dá robustez ao teu artigo
st.subheader("1. Validação Robusta (10-Fold Cross-Validation)")

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

# Configuração da validação cruzada
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(modelo, X, y, cv=kf, scoring='neg_mean_absolute_error')
mae_scores = -scores # Converter para positivo

# Baseline (Modelo "Burro" para comparação)
dummy = DummyRegressor(strategy="mean")
dummy_scores = cross_val_score(dummy, X, y, cv=kf, scoring='neg_mean_absolute_error')
dummy_mae = -dummy_scores.mean()

col1, col2, col3 = st.columns(3)
col1.metric("MAE Médio (O teu Modelo)", f"{mae_scores.mean():.3f}", f"± {mae_scores.std():.3f}")
col2.metric("MAE Baseline (Média Simples)", f"{dummy_mae:.3f}", delta_color="inverse")
col3.metric("Melhoria vs Baseline", f"{((dummy_mae - mae_scores.mean()) / dummy_mae * 100):.1f}%")

st.info("💡 **Para o Artigo:** Se o desvio padrão (±) for baixo, o modelo é estável. Se a melhoria vs Baseline for positiva, o modelo aprendeu algo útil.")

# --- 2. TREINO FINAL E TESTE ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- 3. EXPLICABILIDADE (SHAP) ---
# Apenas para modelos compatíveis (Árvores)
st.subheader("2. Interpretabilidade (Por que razão o modelo decide assim?)")

if modelo_nome in ["XGBoost", "Random Forest", "Gradient Boosting"]:
    st.write("O gráfico abaixo mostra qual variável subjetiva tem mais impacto na previsão automática.")
    
    # Calcular SHAP values
    explainer = shap.Explainer(modelo, X_train)
    shap_values = explainer(X_test)
    
    # Plot SHAP
    fig_shap, ax_shap = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig_shap)
    
    st.caption("Eixo X: Impacto na previsão. Cores: Valor da variável (Vermelho = Alto, Azul = Baixo).")
else:
    st.warning("A análise SHAP (detalhada) só está disponível para XGBoost, Random Forest e Gradient Boosting.")

# --- 4. GRÁFICO REAL vs PREVISTO ---
st.subheader("3. Dispersão: Real vs Previsto")
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='w')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label="Perfeito")
ax.set_xlabel("Valor Real (FaceReader)")
ax.set_ylabel("Previsto pelo Modelo")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)


