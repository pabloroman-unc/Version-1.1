import streamlit as st
import pandas as pd
import numpy as np
import datetime
import hmm_utils

st.set_page_config(page_title="HMM Daily Signals", layout="wide")

st.title("📡 Señales Diarias HMM (DCA)")
st.markdown("""
**Panel de Control Diario:**
Consulta el estado actual del mercado para tus activos.
*   **BUY**: Régimen Bull detectado. Es seguro invertir tu aporte.
*   **WAIT**: Régimen de alta volatilidad/Bear. Mantén tu aporte en Cash (Tasa Libre de Riesgo).
""")

# --- CONFIGURACIÓN ---
st.sidebar.header("Configuración")
tickers_input = st.sidebar.text_area("Tickers (separados por coma)", value="SPY, QQQ, AMD, BRKB, URA, HUT, EEM, IBIT, MELI, META, MSFT, NVDA, SATL, KO, VEA, VIST, XP")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]



# Configuración del Modelo
st.sidebar.subheader("Parámetros del Modelo")
# Usamos una ventana móvil larga por defecto para entrenar
lookback_years = st.sidebar.slider("Años de Historia para Entrenamiento", 5, 20, 15)
n_states = st.sidebar.number_input("Número de Estados", min_value=2, max_value=5, value=3)

if st.sidebar.button("Analizar Mercado"):
    results = []
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=lookback_years*365)
    
    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(tickers):
        try:
            # 1. Fetch Data (Hasta hoy)
            data = hmm_utils.get_data(ticker, start_date, end_date + datetime.timedelta(days=1))
            
            if len(data) < 100:
                st.warning(f"{ticker}: Datos insuficientes.")
                continue
            
            # 2. Train Model
            model, states, feature_cols = hmm_utils.train_hmm(data, n_states)
            bull_state, summary = hmm_utils.get_bull_state(data, states, n_states)
            
            # 3. Get Latest State
            last_state = states[-1]
            last_date = data.index[-1].date()
            last_price = data['Adj Close'].iloc[-1]
            
            # 4. Determine Signal
            signal = "BUY" if last_state == bull_state else "WAIT"
            
            results.append({
                "Ticker": ticker,
                "Fecha": last_date,
                "Precio": f"${last_price:.2f}",
                "Estado Actual": last_state,
                "Estado Bull": bull_state,
                "Señal": signal
            })
            
        except Exception as e:
            st.error(f"Error analizando {ticker}: {e}")
        
        progress_bar.progress((idx + 1) / len(tickers))
    
    # --- RESULTADOS ---
    if results:
        df_results = pd.DataFrame(results)
        
        # Styling
        def color_signal(val):
            color = '#00CC96' if val == 'BUY' else '#EF553B'
            return f'background-color: {color}; color: white; font-weight: bold'
        
        st.subheader("📋 Tablero de Señales")
        st.dataframe(df_results.style.applymap(color_signal, subset=['Señal']))
        

    else:
        st.warning("No se pudieron generar resultados.")
