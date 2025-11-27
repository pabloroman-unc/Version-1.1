import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import hmm_utils

st.set_page_config(page_title="HMM Multi-Ticker Backtest", layout="wide")

st.title("🚀 Backtesting Multi-Ticker")
st.markdown("""
**Comparación de Estrategias:**
Compara el rendimiento de la estrategia HMM aplicada a múltiples activos frente al Buy & Hold del SPY.
""")

# --- CONFIGURACIÓN ---
st.sidebar.header("Configuración")
tickers_input = st.sidebar.text_area("Tickers (separados por coma)", value="SPY, QQQ, IWM")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Fechas
st.sidebar.subheader("Período de Entrenamiento")
train_start = st.sidebar.date_input("Inicio Entrenamiento", datetime.date(2005, 1, 1))
train_end = st.sidebar.date_input("Fin Entrenamiento", datetime.date(2023, 12, 31))

st.sidebar.subheader("Período de Simulación")
test_start = st.sidebar.date_input("Inicio Simulación", datetime.date(2024, 1, 1))
test_end = st.sidebar.date_input("Fin Simulación", datetime.date.today())

n_states = st.sidebar.number_input("Número de Estados", min_value=2, max_value=5, value=3)

if st.sidebar.button("Ejecutar Backtest"):
    try:
        # 1. Obtener Benchmark (SPY)
        full_start = min(train_start, test_start)
        full_end = max(train_end, test_end) + datetime.timedelta(days=1)
        
        with st.spinner("Obteniendo datos de Benchmark (SPY)..."):
            spy_data = hmm_utils.get_data("SPY", full_start, full_end)
            
            # Filtrar para test
            test_mask_spy = (spy_data.index.date >= test_start) & (spy_data.index.date <= test_end)
            spy_test = spy_data[test_mask_spy].copy()
            
            if len(spy_test) == 0:
                st.error("No hay datos de SPY para el período de simulación.")
                st.stop()
                
            spy_test['Cumulative_Market'] = np.exp(spy_test['Returns'].cumsum())
            spy_return = (spy_test['Cumulative_Market'].iloc[-1] - 1) * 100

        # Inicializar gráfico
        fig = go.Figure()
        
        # Plot Benchmark
        fig.add_trace(go.Scatter(
            x=spy_test.index, y=spy_test['Cumulative_Market'],
            mode='lines', name=f'SPY Buy & Hold ({spy_return:.2f}%)',
            line=dict(color='gray', width=2, dash='dot')
        ))
        
        results_summary = []
        results_summary.append({"Ticker": "SPY (Benchmark)", "Estrategia": "Buy & Hold", "Retorno Total": f"{spy_return:.2f}%"})

        # 2. Loop Tickers
        progress_bar = st.progress(0)
        
        for idx, ticker in enumerate(tickers):
            with st.spinner(f"Procesando {ticker}..."):
                try:
                    # Fetch Data
                    data = hmm_utils.get_data(ticker, full_start, full_end)
                    
                    # Split
                    data_idx_date = data.index.date
                    train_mask = (data_idx_date >= train_start) & (data_idx_date <= train_end)
                    test_mask = (data_idx_date >= test_start) & (data_idx_date <= test_end)
                    
                    train_data = data[train_mask].copy()
                    test_data = data[test_mask].copy()
                    
                    if len(train_data) < 50 or len(test_data) == 0:
                        st.warning(f"{ticker}: Datos insuficientes.")
                        continue
                        
                    # Train
                    model, train_states, feature_cols = hmm_utils.train_hmm(train_data, n_states)
                    bull_state, _ = hmm_utils.get_bull_state(train_data, train_states, n_states)
                    
                    # Predict
                    X_test = test_data[feature_cols].values
                    test_states = model.predict(X_test)
                    test_data['State'] = test_states
                    
                    # Strategy
                    test_data['Signal'] = np.where(test_data['State'] == bull_state, 1, 0)
                    test_data['Strategy_Returns'] = test_data['Signal'].shift(1) * test_data['Returns']
                    test_data['Cumulative_Strategy'] = np.exp(test_data['Strategy_Returns'].cumsum())
                    
                    total_ret = (test_data['Cumulative_Strategy'].iloc[-1] - 1) * 100
                    
                    # Trades
                    trades = test_data['Signal'].diff().fillna(0)
                    n_buys = (trades == 1).sum()
                    n_sells = (trades == -1).sum()
                    
                    # Plot
                    fig.add_trace(go.Scatter(
                        x=test_data.index, y=test_data['Cumulative_Strategy'],
                        mode='lines', name=f'{ticker} HMM ({total_ret:.2f}%)'
                    ))
                    
                    results_summary.append({
                        "Ticker": ticker, 
                        "Estrategia": "HMM Rolling Vol", 
                        "Retorno Total": f"{total_ret:.2f}%",
                        "Buys": n_buys,
                        "Sells": n_sells
                    })
                    
                except Exception as e:
                    st.error(f"Error procesando {ticker}: {e}")
            
            progress_bar.progress((idx + 1) / len(tickers))

        # Finalizar
        st.subheader("Resultados Comparativos")
        st.dataframe(pd.DataFrame(results_summary))
        
        fig.update_layout(title="Comparación de Curvas de Equidad", 
                          yaxis_title="Retorno Acumulado", template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error general: {e}")
