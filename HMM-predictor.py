import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import plotly.graph_objects as go

st.set_page_config(page_title="HMM Rolling Regime", layout="wide")

st.title("🧠 HMM con Memoria (Rolling Volatility)")
st.markdown("""
**Estrategia Mejorada para Índices (SPY, QQQ):**
En lugar de reaccionar al ruido diario, este modelo mira el **Régimen de Volatilidad de 20 días**.
* Objetivo: Mantenerse comprado durante correcciones cortas, salir solo en cambios de tendencia estructurales.
""")

    import datetime

    # --- 1. CONFIGURACIÓN ---
    st.sidebar.header("Configuración")
    ticker = st.sidebar.text_input("Ticker", value="SPY")

# Fechas de Entrenamiento
st.sidebar.subheader("Período de Entrenamiento")
train_start = st.sidebar.date_input("Inicio Entrenamiento", datetime.date(2005, 1, 1))
train_end = st.sidebar.date_input("Fin Entrenamiento", datetime.date(2023, 12, 31))

# Fechas de Simulación (Backtest)
st.sidebar.subheader("Período de Simulación")
test_start = st.sidebar.date_input("Inicio Simulación", datetime.date(2024, 1, 1))
test_end = st.sidebar.date_input("Fin Simulación", datetime.date.today())

n_states = 3

# --- 2. OBTENCIÓN DE DATOS ---
@st.cache_data
def get_data(ticker, start, end):
    # Descargamos desde la fecha más antigua hasta la más reciente necesaria
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if 'Adj Close' not in data.columns:
        data['Adj Close'] = data['Close'] if 'Close' in data.columns else None
    
    data.dropna(inplace=True)

    # --- INGENIERÍA DE FEATURES (CLAVE) ---
    # 1. Retornos Diarios
    data['Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    
    # 2. Volatilidad Móvil (Rolling Std Dev) de 20 días
    # Esto mide el "régimen" del mes, filtrando el ruido de un solo día.
    data['Vol_20d'] = data['Returns'].rolling(window=20).std()
    
    # 3. Momentum (RSI Simplificado o Retorno acumulado corto)
    # Añadimos una media móvil simple de 10 días vs precio para detectar tendencia
    data['MA_10'] = data['Adj Close'].rolling(window=10).mean()
    data['Trend'] = np.log(data['Adj Close'] / data['MA_10'])

    data.dropna(inplace=True)
    return data

try:
    # Buscamos datos cubriendo todo el rango seleccionado
    full_start = min(train_start, test_start)
    full_end = max(train_end, test_end) + datetime.timedelta(days=1) # +1 para incluir el día final
    data = get_data(ticker, full_start, full_end)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- 3. PREPROCESAMIENTO ---
# Convertir índices a date para comparar con date_input
data_idx_date = data.index.date

train_mask = (data_idx_date >= train_start) & (data_idx_date <= train_end)
test_mask = (data_idx_date >= test_start) & (data_idx_date <= test_end)

train_data = data[train_mask].copy()
test_data = data[test_mask].copy()

if len(test_data) == 0:
    st.stop()

# Usamos 3 dimensiones ahora: Retorno, Volatilidad del Mes, y Tendencia Corta
feature_cols = ['Returns', 'Vol_20d', 'Trend']
X_train = train_data[feature_cols].values

# --- 4. ENTRENAMIENTO ---
model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
model.fit(X_train)

# --- 5. SELECCIÓN INTELIGENTE DE ESTADO ---
train_states = model.predict(X_train)
train_data['State'] = train_states

summary = []
best_score = -999
bull_state = 0

for i in range(n_states):
    mask = train_data['State'] == i
    avg_ret = train_data.loc[mask, 'Returns'].mean()
    avg_vol = train_data.loc[mask, 'Returns'].std()
    
    # Score Personalizado: Queremos Retorno Positivo Y Baja Volatilidad
    # Penalizamos mucho la volatilidad para evitar el "ruido"
    # Score = Media / (Volatilidad^2)
    score = (avg_ret * 252) / (avg_vol * np.sqrt(252)) if avg_vol > 0 else 0
    
    if score > best_score:
        best_score = score
        bull_state = i
    
    summary.append({
        "Estado": i,
        "Retorno Anualizado": avg_ret * 252,
        "Volatilidad Anual": avg_vol * np.sqrt(252),
        "Score (Sharpe Aprox)": score
    })

st.sidebar.markdown("### 📊 Regímenes Detectados")
st.sidebar.dataframe(pd.DataFrame(summary).style.format("{:.4f}"))
st.sidebar.success(f"Estado Seleccionado (Bull): {bull_state}")

# --- 6. BACKTESTING ---
X_test = test_data[feature_cols].values
test_states = model.predict(X_test)
test_data['State'] = test_states

# Señal
test_data['Signal'] = np.where(test_data['State'] == bull_state, 1, 0)

# Estrategia
test_data['Strategy_Returns'] = test_data['Signal'].shift(1) * test_data['Returns']

# Equidad
test_data['Cumulative_Market'] = np.exp(test_data['Returns'].cumsum())
test_data['Cumulative_Strategy'] = np.exp(test_data['Strategy_Returns'].cumsum())

# --- 7. RESULTADOS ---
total_return_market = (test_data['Cumulative_Market'].iloc[-1] - 1) * 100
total_return_strategy = (test_data['Cumulative_Strategy'].iloc[-1] - 1) * 100

st.subheader("Comparación de Rendimiento (SPY)")
col1, col2 = st.columns(2)
col1.metric("Buy & Hold (SPY)", f"{total_return_market:.2f}%")
col2.metric("HMM Rolling Vol", f"{total_return_strategy:.2f}%", 
            delta=f"{total_return_strategy - total_return_market:.2f}%")

# Gráfico
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=test_data.index, y=test_data['Cumulative_Market'],
    mode='lines', name='Buy & Hold',
    line=dict(color='gray', width=1, dash='dot')
))

fig.add_trace(go.Scatter(
    x=test_data.index, y=test_data['Cumulative_Strategy'],
    mode='lines', name='HMM Estrategia',
    line=dict(color='#00CC96', width=2)
))

# Colorear zonas de compra
shapes = []
current_signal = 0
start_shape = None

for date, row in test_data.iterrows():
    if row['Signal'] == 1 and current_signal == 0:
        start_shape = date
        current_signal = 1
    elif row['Signal'] == 0 and current_signal == 1:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=start_shape, x1=date, y0=0, y1=1,
            fillcolor="rgba(0, 204, 150, 0.1)", line_width=0
        ))
        current_signal = 0
if current_signal == 1:
    shapes.append(dict(
        type="rect", xref="x", yref="paper",
        x0=start_shape, x1=test_data.index[-1], y0=0, y1=1,
        fillcolor="rgba(0, 204, 150, 0.1)", line_width=0
    ))

fig.update_layout(title="Curva de Equidad con Filtro de Volatilidad Móvil", 
                  yaxis_title="Retorno Acumulado", shapes=shapes, template="plotly_dark", height=500)

st.plotly_chart(fig, use_container_width=True)

st.info("Nota: Este modelo intenta mantenerse invertido durante la 'calma', aunque el precio baje un poco, y solo salir cuando la volatilidad estructural de 20 días aumenta.")