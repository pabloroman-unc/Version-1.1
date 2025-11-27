import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import streamlit as st

# --- OBTENCIÓN DE DATOS ---
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

def train_hmm(data, n_states):
    feature_cols = ['Returns', 'Vol_20d', 'Trend']
    X = data[feature_cols].values
    
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
    model.fit(X)
    
    states = model.predict(X)
    return model, states, feature_cols

def get_bull_state(data, states, n_states):
    data_w_states = data.copy()
    data_w_states['State'] = states
    
    summary = []
    best_score = -999
    bull_state = 0
    
    for i in range(n_states):
        mask = data_w_states['State'] == i
        avg_ret = data_w_states.loc[mask, 'Returns'].mean()
        avg_vol = data_w_states.loc[mask, 'Returns'].std()
        
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
        
    return bull_state, summary
