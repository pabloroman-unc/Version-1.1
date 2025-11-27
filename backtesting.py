import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import hmm_utils

st.set_page_config(page_title="HMM Backtesting", layout="wide")

st.title("📈 Backtesting de Estrategia HMM")
st.markdown("""
**Simulación de Estrategia:**
Entrena el modelo en un período histórico y prueba su desempeño en datos fuera de la muestra (Test Set).
""")

# --- CONFIGURACIÓN ---
st.sidebar.header("Configuración del Backtest")
ticker = st.sidebar.text_input("Ticker", value="SPY")

# Fechas de Entrenamiento
st.sidebar.subheader("Período de Entrenamiento (In-Sample)")
train_start = st.sidebar.date_input("Inicio Entrenamiento", datetime.date(2005, 1, 1))
train_end = st.sidebar.date_input("Fin Entrenamiento", datetime.date(2023, 12, 31))

# Fechas de Simulación (Out-of-Sample)
st.sidebar.subheader("Período de Simulación (Out-of-Sample)")
test_start = st.sidebar.date_input("Inicio Simulación", datetime.date(2024, 1, 1))
test_end = st.sidebar.date_input("Fin Simulación", datetime.date.today())

n_states = st.sidebar.number_input("Número de Estados", min_value=2, max_value=5, value=3)

# --- EJECUCIÓN ---
try:
    # Buscamos datos cubriendo todo el rango seleccionado
    full_start = min(train_start, test_start)
    full_end = max(train_end, test_end) + datetime.timedelta(days=1)
    
    data = hmm_utils.get_data(ticker, full_start, full_end)
    
    # --- PREPROCESAMIENTO ---
    data_idx_date = data.index.date
    train_mask = (data_idx_date >= train_start) & (data_idx_date <= train_end)
    test_mask = (data_idx_date >= test_start) & (data_idx_date <= test_end)
    
    train_data = data[train_mask].copy()
    test_data = data[test_mask].copy()
    
    if len(train_data) < 50:
        st.error("Datos de entrenamiento insuficientes.")
        st.stop()
        
    if len(test_data) == 0:
        st.error("No hay datos para el período de simulación seleccionado.")
        st.stop()

    # --- ENTRENAMIENTO ---
    model, train_states, feature_cols = hmm_utils.train_hmm(train_data, n_states)
    
    # Identificar estado Bull en entrenamiento
    bull_state, summary = hmm_utils.get_bull_state(train_data, train_states, n_states)
    
    st.sidebar.success(f"Estado Bull Identificado: {bull_state}")
    st.sidebar.dataframe(pd.DataFrame(summary).style.format("{:.4f}"))

    # --- BACKTESTING ---
    X_test = test_data[feature_cols].values
    test_states = model.predict(X_test)
    test_data['State'] = test_states
    
    # Señal
    test_data['Signal'] = np.where(test_data['State'] == bull_state, 1, 0)
    
    # Estrategia (Shift 1 día)
    test_data['Strategy_Returns'] = test_data['Signal'].shift(1) * test_data['Returns']
    
    # Equidad
    test_data['Cumulative_Market'] = np.exp(test_data['Returns'].cumsum())
    test_data['Cumulative_Strategy'] = np.exp(test_data['Strategy_Returns'].cumsum())
    
    # --- RESULTADOS ---
    total_return_market = (test_data['Cumulative_Market'].iloc[-1] - 1) * 100
    total_return_strategy = (test_data['Cumulative_Strategy'].iloc[-1] - 1) * 100
    
    st.subheader("Comparación de Rendimiento")
    col1, col2 = st.columns(2)
    col1.metric("Buy & Hold", f"{total_return_market:.2f}%")
    col2.metric("HMM Strategy", f"{total_return_strategy:.2f}%", 
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
    
    fig.update_layout(title="Curva de Equidad (Out-of-Sample)", 
                      yaxis_title="Retorno Acumulado", shapes=shapes, template="plotly_dark", height=500)
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error durante el backtest: {e}")
