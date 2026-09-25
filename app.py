import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Dynamic RS Breakout Dashboard", layout="wide")

st.title("🚀 Multi-Timeframe Breakout Dashboard")
st.write("Identifies stocks entering the Green Zone based on composite daily returns (20, 60, 80, 90 days).")

# ==========================================
# 1. SIDEBAR FILE UPLOADER
# ==========================================
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your sector CSV file", type=["csv"])

if not uploaded_file:
    st.info("👆 Please drag and drop your CSV file into the sidebar to get started.")
    st.stop()

# ==========================================
# 2. INGEST DATA FROM UPLOADED CSV
# ==========================================
@st.cache_data
def load_csv_data(file):
    try:
        df = pd.read_csv(file)
        if 'sector' in df.columns:
            df['sector'] = df['sector'].astype(str).str.strip().str.title()
        if 'Symbol' in df.columns:
            df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
        if 'marketcapname' in df.columns:
            df['marketcapname'] = df['marketcapname'].astype(str).str.strip().str.title()
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

master_df = load_csv_data(uploaded_file)

if master_df.empty or 'Symbol' not in master_df.columns:
    st.error('🚨 Could not read the data. Please ensure it is a valid CSV file with a "Symbol" column.')
    st.stop()

# ==========================================
# 3. SIDEBAR FILTERS 
# ==========================================
st.sidebar.header("2. Dashboard Configuration")

if 'sector' in master_df.columns:
    clean_sectors = master_df['sector'].dropna().astype(str).unique()
    available_sectors = sorted(clean_sectors)
    selected_sector = st.sidebar.selectbox("Select Sector to Analyze", available_sectors)
    filtered_df = master_df[master_df['sector'] == selected_sector]
else:
    selected_sector = "All Data"
    filtered_df = master_df

if 'marketcapname' in master_df.columns:
    clean_mcaps = master_df['marketcapname'].dropna().astype(str).unique()
    available_mcaps = ["All"] + sorted(clean_mcaps)
    selected_mcap = st.sidebar.selectbox("Filter by Market Cap", available_mcaps)
    if selected_mcap != "All":
        filtered_df = filtered_df[filtered_df['marketcapname'] == selected_mcap]
else:
    selected_mcap = "All"

active_symbols = filtered_df['Symbol'].tolist()

if not active_symbols:
    st.warning("No stocks found for the selected filters.")
    st.stop()

# ==========================================
# 4. ENGINE (CALCULATING TODAY & 5 DAYS AGO)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_market_data(symbols):
    yf_symbols = [f"{sym}.NS" for sym in symbols]
    
    # Need 255 days to calculate a 250-day window shifted 5 days back
    hist = yf.download(yf_symbols, period="2y", interval="1d", progress=False)
    
    if "Close" in hist:
        closes = hist["Close"]
    else:
        closes = hist
        
    closes = closes.ffill().bfill()
    data = []
    
    for sym, yf_sym in zip(symbols, yf_symbols):
        if yf_sym in closes.columns:
            series = closes[yf_sym].dropna()
            
            if len(series) >= 255:
                current_price = series.iloc[-1]
                daily_returns = series.pct_change() * 100
                
                # --- TODAY'S COMPOSITE ---
                sum_20 = daily_returns.iloc[-20:].sum()
                sum_60 = daily_returns.iloc[-80:-20].sum()
                sum_80 = daily_returns.iloc[-160:-80].sum()
                sum_90 = daily_returns.iloc[-250:-160].sum()
                composite_today = sum_20 + sum_60 + sum_80 + sum_90
                
                # --- 1 WEEK AGO (5 TRADING DAYS) COMPOSITE ---
                prev_20 = daily_returns.iloc[-25:-5].sum()
                prev_60 = daily_returns.iloc[-85:-25].sum()
                prev_80 = daily_returns.iloc[-165:-85].sum()
                prev_90 = daily_returns.iloc[-255:-165].sum()
                composite_prev = prev_20 + prev_60 + prev_80 + prev_90
                
                data.append({
                    "Symbol": sym, 
                    "Current Price (₹)": current_price,
                    "Composite RS Today": composite_today,
                    "Composite RS Prev": composite_prev,
                    "Sum 20D": sum_20,
                    "Sum 60D": sum_60,
                    "Sum 80D": sum_80,
                    "Sum 90D": sum_90
                })
                
    return pd.DataFrame(data)

# ==========================================
# 5. DASHBOARD RENDERING & SIGNAL LOGIC
# ==========================================
with st.spinner(f"Analyzing {len(active_symbols)} stocks for breakouts..."):
    df = fetch_market_data(active_symbols)

if not df.empty:
    st.subheader(f"{selected_sector} Rankings ({selected_mcap})")
    
    # Rank stocks by Percentile (Lower percentile = Top rank)
    df["Today Pct"] = df["Composite RS Today"].rank(pct=True, ascending=False)
    df["Prev Pct"] = df["Composite RS Prev"].rank(pct=True, ascending=False)
    
    # Min-Max Normalization to 0-100 Score for Display
    min_rs = df["Composite RS Today"].min()
    max_rs = df["Composite RS Today"].max()
    df["Score"] = ((df["Composite RS Today"] - min_rs) / (max_rs - min_rs)) * 100 if max_rs != min_rs else 50.0

    # 🚀 BREAKOUT SIGNAL LOGIC
    def get_signal(row):
        # Was in Red (Bottom 35%), now in Green (Top 35%)
        if row["Prev Pct"] >= 0.65 and row["Today Pct"] <= 0.35:
            return "🚀 Epic Breakout (Red to Green)"
        # Was in Gray (Middle), now in Green (Top 35%)
        elif row["Prev Pct"] > 0.35 and row["Today Pct"] <= 0.35:
            return "🔥 Entered Green Zone"
        # Was in Green, dropped out
        elif row["Prev Pct"] <= 0.35 and row["Today Pct"] > 0.35:
            return "⚠️ Exited Green Zone"
        else:
            return "-"

    df["Signal"] = df.apply(get_signal, axis=1)

    if 'Stock Name' in master_df.columns:
        df = df.merge(master_df[['Symbol', 'Stock Name']], on='Symbol', how='left')
        display_cols = ["Signal", "Symbol", "Stock Name", "Current Price (₹)", "Sum 20D", "Composite RS Today", "Score"]
    else:
        display_cols = ["Signal", "Symbol", "Current Price (₹)", "Sum 20D", "Composite RS Today", "Score"]

    # Sort by the final Score
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  
    df.index.name = "Rank"
    
    # Clean up numbers
    for col in ["Current Price (₹)", "Sum 20D", "Composite RS Today", "Score"]:
        df[col] = df[col].round(2)
        
    display_df = df[[col for col in display_cols if col in df.columns]]
    
    # Color Heatmap Logic based on Today's Percentile
    def apply_color_ranking(data):
        active_count = len(data)
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        
        for i in range(active_count):
            rank = i + 1
            pct = rank / active_count
            
            if pct <= 0.35:
                color = "background-color: rgba(0, 128, 0, 0.4); color: white;"
            elif pct >= 0.65:
                color = "background-color: rgba(255, 0, 0, 0.4); color: white;"
            else:
                color = "background-color: rgba(128, 128, 128, 0.4); color: white;"
                
            styles.iloc[i] = color
        return styles

    styled_df = display_df.style.apply(apply_color_ranking, axis=None)

    st.dataframe(styled_df, use_container_width=True, height=800)
else:
    st.warning("Failed to retrieve sufficient market data.")
