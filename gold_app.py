import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import pytz

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Gold Intraday Predictor", layout="wide")

st.title("🏆 Aplikasi Ramalan Emas Intraday (XAU/USD)")
st.markdown("Analisis teknikal & ramalan trend mudah untuk pasaran Emas.")

# --- Sidebar untuk Input ---
st.sidebar.header("Tetapan Parameter")
interval = st.sidebar.selectbox("Pilih Timeframe:", ["15m", "30m", "1h", "1d"], index=0)
period = "5d" if interval in ["15m", "30m", "1h"] else "1mo"

# --- Fungsi Tarik Data ---
def get_gold_data(interval, period):
    # Menggunakan GC=F (Gold Futures) sebagai proksi paling dekat dengan XAU/USD di Yahoo Finance
    ticker = "GC=F" 
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if data.empty:
        st.error("Gagal mendapatkan data. Pasaran mungkin tutup (Sabtu/Ahad).")
        return None
    
    data.reset_index(inplace=True)
    
    # Bersihkan data MultiIndex jika ada (format baru yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
        
    # Tukar zon masa ke Malaysia (KL Time)
    kl_tz = pytz.timezone('Asia/Kuala_Lumpur')
    if data['Datetime'].dt.tz is None:
         # Jika tiada timezone, anggap UTC dulu kemudian convert
         data['Datetime'] = data['Datetime'].dt.tz_localize('UTC').dt.tz_convert(kl_tz)
    else:
         data['Datetime'] = data['Datetime'].dt.tz_convert(kl_tz)
         
    return data

# --- Fungsi Kira Indikator ---
def add_indicators(df):
    # RSI
    rsi_indicator = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()
    
    # MACD
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    
    # SMA (Moving Average)
    df["SMA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    
    return df

# --- Fungsi Machine Learning Mudah (Linear Regression) ---
def predict_next_price(df):
    # Kita guna data 50 candle terakhir untuk train model ringkas
    df_train = df.tail(50).copy()
    df_train = df_train.dropna()
    
    if len(df_train) < 10:
        return 0, "Data tidak mencukupi"

    X = np.array(range(len(df_train))).reshape(-1, 1)
    y = df_train['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Ramal candle seterusnya (index seterusnya)
    next_index = np.array([[len(df_train)]])
    prediction = model.predict(next_index)
    
    current_price = df_train['Close'].iloc[-1]
    trend = "NAIK 📈" if prediction[0] > current_price else "TURUN 📉"
    
    return prediction[0], trend

# --- Main App Logic ---
data = get_gold_data(interval, period)

if data is not None:
    data = add_indicators(data)
    
    # Paparan Harga Terkini
    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    price_change = last_row['Close'] - prev_row['Close']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Terkini (USD)", f"${last_row['Close']:.2f}", f"{price_change:.2f}")
    col2.metric("RSI (14)", f"{last_row['RSI']:.2f}", "Overbought > 70 | Oversold < 30")
    col3.metric("Volume", f"{last_row['Volume']:,}")

    # --- Signal Dashboard ---
    st.subheader("📢 Signal Analisis Teknikal")
    
    signal_col1, signal_col2 = st.columns(2)
    
    # Logic Signal Mudah
    signal = "NEUTRAL 😐"
    color = "grey"
    
    if last_row['RSI'] < 30:
        signal = "STRONG BUY (Oversold) 🟢"
        color = "green"
    elif last_row['RSI'] > 70:
        signal = "STRONG SELL (Overbought) 🔴"
        color = "red"
    elif last_row['MACD'] > last_row['MACD_Signal']:
        signal = "BUY (MACD Cross) 🟢"
        color = "green"
    elif last_row['MACD'] < last_row['MACD_Signal']:
        signal = "SELL (MACD Cross) 🔴"
        color = "red"
        
    with signal_col1:
        st.info(f"Signal Teknikal Semasa: **{signal}**")
        
    # ML Prediction
    predicted_price, trend_direction = predict_next_price(data)
    with signal_col2:
        st.warning(f"Ramalan AI (Next Candle): **${predicted_price:.2f}** ({trend_direction})")

    # --- Graf Candlestick ---
    st.subheader(f"Carta Harga Emas ({interval})")
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=data['Datetime'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Harga Emas'))
    
    # Tambah SMA line
    fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # --- Raw Data ---
    with st.expander("Lihat Data Mentah"):
        st.dataframe(data.sort_values(by="Datetime", ascending=False))

else:
    st.write("Sila cuba lagi nanti.")
