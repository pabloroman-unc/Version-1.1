import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
import hmm_utils

st.set_page_config(page_title="HMM Model Analysis", layout="wide")

st.title("🧠 Análisis del Modelo HMM")
st.markdown("""
**Herramienta de Detección de Regímenes:**
Utiliza esta herramienta para ajustar los parámetros del modelo y visualizar los regímenes de mercado detectados.
""")

# --- CONFIGURACIÓN ---
st.sidebar.header("Configuración del Modelo")
ticker = st.sidebar.text_input("Ticker", value="SPY")

st.sidebar.subheader("Período de Entrenamiento")
train_start = st.sidebar.date_input("Inicio Entrenamiento", datetime.date(2005, 1, 1))
train_end = st.sidebar.date_input("Fin Entrenamiento", datetime.date(2023, 12, 31))

n_states = st.sidebar.number_input("Número de Estados", min_value=2, max_value=5, value=3)

# --- EJECUCIÓN ---
try:
    # +1 día para incluir el último día en la descarga
    end_date_fetch = train_end + datetime.timedelta(days=1)
    data = hmm_utils.get_data(ticker, train_start, end_date_fetch)
    
    # Filtrar exactamente por el rango seleccionado (por si yfinance trae algo más)
    mask = (data.index.date >= train_start) & (data.index.date <= train_end)
    data = data[mask].copy()
    
    if len(data) < 50:
        st.error("No hay suficientes datos para entrenar.")
        st.stop()

    model, states, feature_cols = hmm_utils.train_hmm(data, n_states)
    data['State'] = states
    
    bull_state, summary = hmm_utils.get_bull_state(data, states, n_states)
    
    # --- VISUALIZACIÓN ---
    st.subheader("Estadísticas de los Estados Detectados")
    st.dataframe(pd.DataFrame(summary).style.format("{:.4f}"))
    st.success(f"Estado Seleccionado como 'Bull' (Alcista): {bull_state}")
    
    st.subheader("Visualización de Regímenes")
    
    # Gráfico de Precio coloreado por Estado
    fig = go.Figure()
    
    # Mapeo de colores para los estados
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'] # Plotly default colors
    
    for i in range(n_states):
        mask = data['State'] == i
        # Para que el gráfico de líneas sea continuo, necesitamos manejar los huecos.
        # Una forma simple es plotear todo en gris de fondo y luego superponer los puntos/tramos.
        # Pero para simplicidad visual, pintaremos puntos o barras.
        
        fig.add_trace(go.Scatter(
            x=data.index[mask], y=data.loc[mask, 'Adj Close'],
            mode='markers', name=f'Estado {i}',
            marker=dict(size=4, color=colors[i % len(colors)])
        ))
        
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Adj Close'],
        mode='lines', name='Precio',
        line=dict(color='gray', width=1),
        opacity=0.5
    ))

    fig.update_layout(title=f"Regímenes de Mercado para {ticker}", yaxis_title="Precio Ajustado", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualización de Features
    st.subheader("Distribución de Features por Estado")
    cols = st.columns(len(feature_cols))
    
    for idx, feature in enumerate(feature_cols):
        with cols[idx]:
            st.write(f"**{feature}**")
            for i in range(n_states):
                subset = data[data['State'] == i][feature]
                st.write(f"Estado {i}: Media {subset.mean():.4f}, Std {subset.std():.4f}")

except Exception as e:
    st.error(f"Error durante el análisis: {e}")
