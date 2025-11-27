import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import hmm_utils

st.set_page_config(page_title="HMM DCA Backtest", layout="wide")

st.title("💰 DCA con Timing HMM")
st.markdown("""
**Estrategia de Aportaciones Periódicas (DCA):**
Simula una inversión de **$1 cada N días**.
*   **Benchmark (SPY)**: Compra SPY inmediatamente (DCA Estándar).
*   **Estrategia**: Acumula el efectivo y solo compra cuando el HMM detecta un régimen **Bull**. Una vez comprado, mantiene (Hold).
""")

# --- CONFIGURACIÓN ---
st.sidebar.header("Configuración")
tickers_input = st.sidebar.text_area("Tickers (separados por coma)", value="SPY, QQQ, IWM")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

contribution_days = st.sidebar.number_input("Días entre Aportaciones", min_value=1, value=10)


# Fechas
st.sidebar.subheader("Período de Simulación")

# ... (Scenario logic remains same, skipping lines for brevity if possible, but replace needs context)
# Actually, I need to be careful not to overwrite the scenario logic.
# Let's target the sidebar input area specifically.

# ... (Rest of file)


scenario = st.sidebar.selectbox(
    "Seleccionar Escenario",
    options=["Custom", "2008 Financial Crisis", "2022 Bear Market", "Dot Com Bubble (2000)"],
    index=0
)

if scenario == "2008 Financial Crisis":
    default_train_start = datetime.date(1995, 1, 1)
    default_train_end = datetime.date(2007, 9, 30)
    default_test_start = datetime.date(2007, 10, 1)
    default_test_end = datetime.date(2010, 1, 1)
elif scenario == "2022 Bear Market":
    default_train_start = datetime.date(2010, 1, 1)
    default_train_end = datetime.date(2021, 12, 31)
    default_test_start = datetime.date(2022, 1, 1)
    default_test_end = datetime.date(2023, 1, 1)
elif scenario == "Dot Com Bubble (2000)":
    default_train_start = datetime.date(1993, 1, 1)
    default_train_end = datetime.date(1999, 12, 31)
    default_test_start = datetime.date(2000, 1, 1)
    default_test_end = datetime.date(2003, 1, 1)
else:
    default_train_start = datetime.date(2005, 1, 1)
    default_train_end = datetime.date(2023, 12, 31)
    default_test_start = datetime.date(2024, 1, 1)
    default_test_end = datetime.date.today()

train_start = st.sidebar.date_input("Inicio Entrenamiento", default_train_start)
train_end = st.sidebar.date_input("Fin Entrenamiento", default_train_end)
test_start = st.sidebar.date_input("Inicio Simulación", default_test_start)
test_end = st.sidebar.date_input("Fin Simulación", default_test_end)

n_states = st.sidebar.number_input("Número de Estados", min_value=2, max_value=5, value=3)

if st.sidebar.button("Ejecutar Backtest"):
    try:
        # 1. Obtener Benchmark (SPY)
        full_start = min(train_start, test_start)
        full_end = max(train_end, test_end) + datetime.timedelta(days=1)
        
        with st.spinner("Simulando Benchmark (SPY DCA)..."):
            spy_data = hmm_utils.get_data("SPY", full_start, full_end)
            
            # Filtrar para test
            test_mask_spy = (spy_data.index.date >= test_start) & (spy_data.index.date <= test_end)
            spy_test = spy_data[test_mask_spy].copy()
            
            if len(spy_test) == 0:
                st.error("No hay datos de SPY para el período de simulación.")
                st.stop()
            
            # --- Lógica DCA Benchmark ---
            spy_test['Cash_Contribution'] = 0.0
            spy_test['Shares'] = 0.0
            spy_test['Portfolio_Value'] = 0.0
            spy_test['Total_Invested'] = 0.0
            
            current_shares = 0.0
            total_invested = 0.0
            
            # Iteramos cada N días para aportar
            # Para eficiencia, podemos usar iloc con step, pero necesitamos rellenar el DF completo
            # Haremos un loop simple para claridad de la lógica de "día de aporte"
            
            contribution_dates = spy_test.index[::contribution_days]
            
            # Vectorizamos la contribución
            spy_test.loc[contribution_dates, 'Cash_Contribution'] = 1.0
            
            # Calculamos shares compradas cada día de aporte
            # Shares = Contribution / Price
            spy_test['New_Shares'] = spy_test['Cash_Contribution'] / spy_test['Adj Close']
            
            # Acumulamos shares
            spy_test['Shares'] = spy_test['New_Shares'].cumsum()
            spy_test['Total_Invested'] = spy_test['Cash_Contribution'].cumsum()
            
            # Valor Portfolio = Shares * Price
            spy_test['Portfolio_Value'] = spy_test['Shares'] * spy_test['Adj Close']
            
            spy_final_value = spy_test['Portfolio_Value'].iloc[-1]
            spy_invested = spy_test['Total_Invested'].iloc[-1]
            spy_return_pct = ((spy_final_value / spy_invested) - 1) * 100 if spy_invested > 0 else 0

        # Inicializar gráfico
        fig = go.Figure()
        
        # Plot Benchmark
        fig.add_trace(go.Scatter(
            x=spy_test.index, y=spy_test['Portfolio_Value'],
            mode='lines', name=f'SPY DCA Estándar (Val: ${spy_final_value:.2f}, Ret: {spy_return_pct:.2f}%)',
            line=dict(color='gray', width=2, dash='dot')
        ))
        
        # Plot Total Invested (Capital)
        fig.add_trace(go.Scatter(
            x=spy_test.index, y=spy_test['Total_Invested'],
            mode='lines', name='Capital Invertido (Acumulado)',
            line=dict(color='white', width=1, dash='dashdot'),
            opacity=0.5
        ))
        
        results_summary = []
        results_summary.append({
            "Ticker": "SPY (Benchmark)", 
            "Estrategia": "DCA Estándar", 
            "Capital Invertido": f"${spy_invested:.2f}",
            "Valor Final": f"${spy_final_value:.2f}",
            "Retorno Total": f"{spy_return_pct:.2f}%"
        })

        # 2. Loop Tickers Strategy
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
                    test_data['Is_Bull'] = (test_data['State'] == bull_state)
                    
                    # --- Lógica DCA Strategy ---
                    # 1. Recibir aporte cada N días
                    # 2. Si es Bull, comprar con TODO el cash acumulado
                    # 3. Si no, acumular cash
                    
                    test_data['Cash_Contribution'] = 0.0
                    test_data.loc[test_data.index.intersection(contribution_dates), 'Cash_Contribution'] = 1.0
                    
                    # Simulación Loop (necesario por el estado del Cash acumulado)
                    cash_pool = 0.0
                    shares_owned = 0.0
                    portfolio_values = []


                    for date, row in test_data.iterrows():
                        # Recibir aporte
                        cash_pool += row['Cash_Contribution']
                        
                        # Decisión de Compra (Usamos señal del día actual para comprar al cierre/ajustado de hoy)
                        # Nota: En backtest real, usaríamos señal de ayer para comprar hoy, o señal de hoy para comprar al cierre.
                        # Asumimos compra al precio 'Adj Close' si la señal es Bull.
                        
                        if row['Is_Bull'] and cash_pool > 0:
                            # Comprar
                            shares_bought = cash_pool / row['Adj Close']
                            shares_owned += shares_bought
                            cash_pool = 0.0
                        
                        # Valor actual
                        val = (shares_owned * row['Adj Close']) + cash_pool
                        portfolio_values.append(val)
                    
                    test_data['Portfolio_Value'] = portfolio_values
                    
                    final_val = test_data['Portfolio_Value'].iloc[-1]
                    # El capital invertido es el mismo para todos si las fechas coinciden
                    # Usamos el de SPY como referencia de "lo que salió del bolsillo"
                    # O calculamos el propio si las fechas difieren ligeramente
                    my_invested = test_data['Cash_Contribution'].sum()
                    
                    ret_pct = ((final_val / my_invested) - 1) * 100 if my_invested > 0 else 0
                    delta_spy = ret_pct - spy_return_pct
                    
                    # Plot
                    fig.add_trace(go.Scatter(
                        x=test_data.index, y=test_data['Portfolio_Value'],
                        mode='lines', name=f'{ticker} DCA HMM (Val: ${final_val:.2f}, Ret: {ret_pct:.2f}%)'
                    ))
                    
                    results_summary.append({
                        "Ticker": ticker, 
                        "Estrategia": "DCA Timing HMM", 
                        "Capital Invertido": f"${my_invested:.2f}",
                        "Valor Final": f"${final_val:.2f}",
                        "Retorno Total": f"{ret_pct:.2f}%",
                        "vs SPY": f"{delta_spy:+.2f}%",

                    })
                    
                except Exception as e:
                    st.error(f"Error procesando {ticker}: {e}")
            
            progress_bar.progress((idx + 1) / len(tickers))

        # Finalizar
        st.subheader("Resultados Comparativos (DCA)")
        st.dataframe(pd.DataFrame(results_summary))
        
        fig.update_layout(title="Evolución del Valor del Portafolio (DCA)", 
                          yaxis_title="Valor ($)", template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error general: {e}")
