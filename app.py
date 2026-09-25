import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Dynamic RS Dashboard", layout="wide")

st.title("📊 Multi-Timeframe CSV Sector Dashboard")
st.write("Ranks stocks using a composite sum of daily returns across 20, 60, 80, and 90-day historical windows.")

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
# 4. CHARTINK-STYLE MULTI-TIMEFRAME ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def fetch_market_data(symbols):
    yf_symbols = [f"{sym}.NS" for sym in symbols]
    
    # 250 days of calculations requires at least 251 historical price candles
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
            
            if len(series) >= 251:
                current_price = series.iloc[-1]
                
                # Calculate daily percentage change for the entire series
                daily_returns = series.pct_change() * 100
                
                # 1. Daily sum over the last 20 days
                sum_20 = daily_returns.iloc[-20:].sum()
                
                # 2. 20 days ago sum for 60 days (days -80 to -21)
                sum_60 = daily_returns.iloc[-80:-20].sum()
                
                # 3. 80 days ago sum for 80 days (days -160 to -81)
                sum_80 = daily_returns.iloc[-160:-80].sum()
                
                # 4. 160 days ago sum for 90 days (days -250 to -161)
                sum_90 = daily_returns.iloc[-250:-160].sum()
                
                # Combine into total composite momentum score
                composite = sum_20 + sum_60 + sum_80 + sum_90
                
                data.append({
                    "Symbol": sym, 
                    "Current Price (₹)": current_price,
                    "Sum 20D": sum_20,
                    "Sum 60D": sum_60,
                    "Sum 80D": sum_80,
                    "Sum 90D": sum_90,
                    "Composite RS": composite
                })
            else:
                pass # Skip stocks that do not have 250 days of trading history
                
    return pd.DataFrame(data)

# ==========================================
# 5. DASHBOARD RENDERING
# ==========================================
with st.spinner(f"Fetching market data and calculating multi-timeframe returns for {len(active_symbols)} stocks..."):
    df = fetch_market_data(active_symbols)

if not df.empty:
    st.subheader(f"{selected_sector} Rankings ({selected_mcap})")
    
    # Min-Max Normalization to 0-100 Score based on the new Composite RS
    min_rs = df["Composite RS"].min()
    max_rs = df["Composite RS"].max()
    
    if max_rs != min_rs:
        df["Score"] = ((df["Composite RS"] - min_rs) / (max_rs - min_rs)) * 100
    else:
        df["Score"] = 50.0

    if 'Stock Name' in master_df.columns:
        df = df.merge(master_df[['Symbol', 'Stock Name']], on='Symbol', how='left')
        display_cols = ["Symbol", "Stock Name", "Current Price (₹)", "Sum 20D", "Sum 60D", "Sum 80D", "Sum 90D", "Composite RS", "Score"]
    else:
        display_cols = ["Symbol", "Current Price (₹)", "Sum 20D", "Sum 60D", "Sum 80D", "Sum 90D", "Composite RS", "Score"]

    # Sort by the final Score
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  
    df.index.name = "Rank"
    
    # Clean up formatting for readability
    for col in ["Current Price (₹)", "Sum 20D", "Sum 60D", "Sum 80D", "Sum 90D", "Composite RS", "Score"]:
        df[col] = df[col].round(2)
        
    display_df = df[[col for col in display_cols if col in df.columns]]
    
    # Color Heatmap Logic (Applied to the final Score)
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
    
    csv = display_df.to_csv(index=True).encode("utf-8")
    st.download_button("Download Rankings as CSV", data=csv, file_name="multi_timeframe_rankings.csv", mime="text/csv")
else:
    st.warning("Failed to retrieve sufficient market data. Stocks may be too new for a 250-day calculation.")
