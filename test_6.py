import os
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging 

# Configure basic logging for simulated alerts
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration & Setup ---
st.set_page_config(page_title="BTC/ETH Combined Index Signal Tracker", layout="wide")

# Initialize session state for persistent signal tracking (for simulating alerts)
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = pd.Timestamp.min.tz_localize(None)

# -----------------------
# Helper function for external alert (SIMULATION)
# -----------------------

def send_external_alert(signal_type: str, message: str, email: str, phone: str):
    """
    Simulates sending an external alert via Email/SMS API.
    """
    if email or phone:
        logging.info(f"*** EXTERNAL ALERT SENT (Simulated) ***")
        if email:
            logging.info(f"EMAIL To: {email}")
        if phone:
            logging.info(f"SMS To: {phone}")
        logging.info(f"CONTENT: {message.replace('\n', ' | ')}")
    else:
        logging.info("External alert skipped: No email or phone recipient configured.")


# -----------------------
# Helper functions for calculations
# -----------------------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Calculates the Exponential Moving Average-based Relative Strength Index (EMA-RSI).
    Uses EMA smoothing for responsiveness.
    """
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    
    # Use EWM (Exponential Moving Average) for smoothing (EMA-RSI)
    ma_up = up.ewm(span=length, adjust=False).mean()
    ma_down = down.ewm(span=length, adjust=False).mean()
    
    # Avoid division by zero
    rs = ma_up / ma_down.replace(0, 1e-10) 
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# -----------------------
# Sidebar / user inputs
# -----------------------
st.sidebar.header("Index & Indicator Settings")
# Fixed Tickers: BTC-USD and ETH-USD
TICKERS = ["BTC-USD", "ETH-USD"]

# Intraday limit enforcement (FIX from earlier conversation)
MAX_INTRADAY_DAYS = 60
period_days = st.sidebar.number_input("Fetch period (days)", min_value=7, max_value=365, value=7) 
interval = st.sidebar.selectbox("Interval", options=["15m","30m","1h","1d"], index=2)

if interval in ["15m", "30m", "1h"] and period_days > MAX_INTRADAY_DAYS:
    period_days = MAX_INTRADAY_DAYS
    st.sidebar.warning(f"Intraday period capped at {MAX_INTRADAY_DAYS} days.")

st.sidebar.markdown("---")
rsi_length = st.sidebar.number_input("RSI length", min_value=7, max_value=30, value=14)
index_ema_span = st.sidebar.number_input("Cumulative Index EMA Span (Smoother)", min_value=1, max_value=10, value=5)
st.sidebar.markdown("_The EMA span above is for smoothing the index before RSI calculation._")
st.sidebar.markdown("---")

# EMA inputs with new EMA 72
ema_short = st.sidebar.number_input("EMA Short (14)", min_value=5, max_value=25, value=14)
ema_long = st.sidebar.number_input("EMA Medium (30)", min_value=26, max_value=50, value=30)
ema_very_long = st.sidebar.number_input("EMA Very Long (72)", min_value=51, max_value=365, value=72) # New 72 EMA

st.sidebar.markdown("---")
min_bars_after_cycle = st.sidebar.number_input("Max bars to look for re-alignment (0 = unlimited)", min_value=0, max_value=9999, value=0)

# Volume Filtering
volume_length = st.sidebar.number_input("Volume MA Length", min_value=1, max_value=50, value=14)
enable_volume_filter = st.sidebar.checkbox("Require Volume Confirmation", value=False)


st.sidebar.header("External Notification Settings")
recipient_email = st.sidebar.text_input("Recipient Email (for simulation)", value="")
recipient_phone = st.sidebar.text_input("Recipient Phone (for simulation, e.g., +15551234)", value="")
st.sidebar.markdown("_The external alerts are simulated via logging. To make them real, you'd integrate a service like Twilio or SendGrid._")
st.markdown("---")

# --- Main Title ---
st.title(f"🔥 BTC/ETH Combined Index (50/50) Tracker")
st.subheader(f"Triple-EMA Confirmation: {ema_short} > {ema_long} > {ema_very_long}")

st.sidebar.markdown("RSI cycle rules: **rising** = cross up 29 → later cross up 71. **falling** = cross down 71 → later cross down 29.")
st.sidebar.markdown("Signals fire only after a completed cycle + Normalized Price dip/spike + **Triple EMA Alignment**.")


# -----------------------
# 🚀 Data Fetching and Processing (Optimized with st.cache_data)
# -----------------------

@st.cache_data(ttl=timedelta(seconds=300))
def fetch_and_process_data(tickers: list, period_days: int, interval: str, index_ema_span: int):
    """Fetches data for multiple tickers, creates the normalized combined index, and calculates indicators."""
    
    status_text = st.empty()
    status_text.info(f"Fetching {', '.join(tickers)} {interval} data for {period_days} days. Cache TTL: 300 seconds.")
    
    end_date = pd.Timestamp.now()
    start_date = end_date - timedelta(days=period_days)
    
    data = {}
    
    try:
        # Fetch data for all tickers
        raw = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False)
    except Exception as e:
        status_text.error(f"Failed during data fetch: {e}. Check ticker symbols or try reducing period.")
        return pd.DataFrame()

    if raw.empty:
        status_text.error("Fetched data is empty. Try a different interval or period.")
        return pd.DataFrame()
        
    # --- 1. CLEANING AND NORMALIZATION ---
    df = pd.DataFrame()
    total_volume = 0
    
    for ticker in tickers:
        # Handle case where yf.download returns a MultiIndex (common for multiple tickers)
        if isinstance(raw['Close'], pd.DataFrame):
            close_raw = raw['Close'][ticker]
            volume_raw = raw['Volume'][ticker]
        else: # Single ticker case for robustness, though TICKERS is a list of two
            close_raw = raw['Close']
            volume_raw = raw['Volume']
        
        # Data cleaning (forward/backward fill)
        close_raw = close_raw.ffill().bfill()
        volume_raw = volume_raw.ffill().bfill()
        
        # Ensure TZN-naive index
        if close_raw.index.tz is not None:
            close_raw.index = close_raw.index.tz_localize(None)

        # Normalized Cumulative Price
        base_price = close_raw.iloc[0]
        if base_price == 0:
            status_text.error(f"Initial price for {ticker} is zero, cannot normalize.")
            return pd.DataFrame()
        
        df[f'{ticker.split("-")[0].lower()}_cum'] = close_raw / base_price
        total_volume += volume_raw

    # --- 2. COMBINED INDEX ---
    # The combined index is the equally weighted average of the normalized prices
    df['index_cum'] = df[[f'{t.split("-")[0].lower()}_cum' for t in tickers]].mean(axis=1)
    
    # Apply Exponential Smoothing (EMA) to the index itself (the requested "Exponential Cumulative Tracker")
    df['index_cum_smooth'] = df['index_cum'].ewm(span=index_ema_span, adjust=False).mean()

    # --- 3. VOLUME AND INDICATORS ---
    df['volume'] = total_volume
    df['Volume_MA'] = df['volume'].rolling(volume_length, min_periods=1).mean()
    
    status_text.empty()
    return df

# --- Execution ---
df = fetch_and_process_data(TICKERS, period_days, interval, index_ema_span)

if df.empty:
    st.stop()

# -----------------------
# Indicators & Cycles (Calculated on the SMOOTHED Combined Index)
# -----------------------
df['EMA_short'] = df['index_cum_smooth'].ewm(span=ema_short, adjust=False).mean()
df['EMA_long'] = df['index_cum_smooth'].ewm(span=ema_long, adjust=False).mean() 
df['EMA_very_long'] = df['index_cum_smooth'].ewm(span=ema_very_long, adjust=False).mean() # New EMA 72
df['RSI'] = rsi(df['index_cum_smooth'], length=rsi_length)


# --- RSI Cycle Detection ---
cycle_id = 0
in_cycle = False
cycle_type = None
cycle_start_idx = None
cycles = [] 
rsi_series = df['RSI']

df_index_list = df.index.to_list() 

if len(df_index_list) > 1:
    prev_rsi = rsi_series.iloc[0]
    for idx in df_index_list[1:]:
        cur_rsi = rsi_series.loc[idx]
        
        # Cycle start detection
        if not in_cycle:
            if (prev_rsi <= 29) and (cur_rsi > 29):
                in_cycle = True; cycle_type = 'rising'; cycle_start_idx = idx; cycle_id += 1
            elif (prev_rsi >= 71) and (cur_rsi < 71):
                in_cycle = True; cycle_type = 'falling'; cycle_start_idx = idx; cycle_id += 1
        # Cycle end detection
        else:
            if cycle_type == 'rising' and (prev_rsi < 71) and (cur_rsi >= 71):
                cycles.append({'id': cycle_id, 'type': 'rising', 'start': cycle_start_idx, 'end': idx})
                in_cycle = False; cycle_type = None; cycle_start_idx = None
            elif cycle_type == 'falling' and (prev_rsi > 29) and (cur_rsi <= 29):
                cycles.append({'id': cycle_id, 'type': 'falling', 'start': cycle_start_idx, 'end': idx})
                in_cycle = False; cycle_type = None; cycle_start_idx = None
        prev_rsi = cur_rsi

# -----------------------
# Realignment detection and signal setting
# -----------------------
df['signal'] = 0 # 1 buy, -1 sell
df['signal_reason'] = None

# Volume check function
def check_volume_confirmation(idx):
    if not enable_volume_filter:
        return True # Filter disabled, always allow signal
    
    # Check if current volume is greater than its average (Volume_MA)
    return df.at[idx, 'volume'] > df.at[idx, 'Volume_MA']

for c in cycles:
    end_idx = c['end']
    search_idx_list = df.loc[end_idx:].index.to_list()
    if len(search_idx_list) <= 1:
        continue
    
    if min_bars_after_cycle > 0:
        search_idx_list = search_idx_list[1:min_bars_after_cycle+2] 
    else:
        search_idx_list = search_idx_list[1:]

    dipped = False; spiked = False
    
    if c['type'] == 'rising':
        dip_idx = None; reclaim_idx = None
        for t in search_idx_list:
            # Look for dip below EMA long
            if (not dipped) and (df.at[t, 'index_cum_smooth'] < df.at[t, 'EMA_long']):
                dipped = True; dip_idx = t
            # Look for reclaim above EMA long
            if dipped and (df.at[t, 'index_cum_smooth'] > df.at[t, 'EMA_long']):
                reclaim_idx = t
                
                # FINAL BUY CONDITIONS: TRIPLE EMA alignment AND optional Volume confirmation
                is_stacked = (df.at[reclaim_idx, 'EMA_short'] > df.at[reclaim_idx, 'EMA_long']) and \
                             (df.at[reclaim_idx, 'EMA_long'] > df.at[reclaim_idx, 'EMA_very_long'])
                             
                if is_stacked and check_volume_confirmation(reclaim_idx):
                    df.at[reclaim_idx, 'signal'] = 1
                    vol_note = " (Vol Confirmed)" if enable_volume_filter else ""
                    df.at[reclaim_idx, 'signal_reason'] = f"Buy: end rising cycle {c['id']} dip@{dip_idx.strftime('%H:%M')} reclaim@{reclaim_idx.strftime('%H:%M')}{vol_note}"
                    break
                else: break
                    
    elif c['type'] == 'falling':
        spike_idx = None; drop_idx = None
        for t in search_idx_list:
            # Look for spike above EMA long
            if (not spiked) and (df.at[t, 'index_cum_smooth'] > df.at[t, 'EMA_long']):
                spiked = True; spike_idx = t
            # Look for drop below EMA long
            if spiked and (df.at[t, 'index_cum_smooth'] < df.at[t, 'EMA_long']):
                drop_idx = t
                
                # FINAL SELL CONDITIONS: TRIPLE EMA alignment AND optional Volume confirmation
                is_stacked = (df.at[drop_idx, 'EMA_short'] < df.at[drop_idx, 'EMA_long']) and \
                             (df.at[drop_idx, 'EMA_long'] < df.at[drop_idx, 'EMA_very_long'])
                             
                if is_stacked and check_volume_confirmation(drop_idx):
                    df.at[drop_idx, 'signal'] = -1
                    vol_note = " (Vol Confirmed)" if enable_volume_filter else ""
                    df.at[drop_idx, 'signal_reason'] = f"Sell: end falling cycle {c['id']} spike@{spike_idx.strftime('%H:%M')} drop@{drop_idx.strftime('%H:%M')}{vol_note}"
                    break
                else: break

# -----------------------
# Real-time Alerting (External + Internal)
# -----------------------
latest_signal = df[df['signal'] != 0].tail(1)

if not latest_signal.empty:
    latest_time = latest_signal.index[0] 
    signal_value = latest_signal['signal'].iloc[0]
    signal_type = "BUY" if signal_value == 1 else "SELL"
    
    if latest_time > st.session_state.last_signal_time:
        st.session_state.last_signal_time = latest_time
        
        # --- 1. Internal Alert Message (in the app) ---
        alert_message = (
            f"🔔 **NEW ALERT ({signal_type})**: Cycle Realignment Signal Fired for BTC/ETH Index!\n\n"
            f"**Time**: {latest_time.strftime('%Y-%m-%d %H:%M:%S')} ({interval})\n"
            f"**Action**: {signal_type}\n"
            f"**Reason**: {latest_signal['signal_reason'].iloc[0]}"
        )
        st.error(alert_message, icon="🚨") 

        # --- 2. External Alert Generation (Simulated) ---
        external_message = f"BTC/ETH Index ALERT ({interval}): {signal_type} at {latest_time.strftime('%H:%M')}. Reason: {latest_signal['signal_reason'].iloc[0]}."
        send_external_alert(signal_type, external_message, recipient_email, recipient_phone)


# -----------------------
# Plotting: main chart (Price/EMAs) + RSI subplot
# -----------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.25], vertical_spacing=0.05,
                    specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

# 1. Price Normalized Tracks (Faded)
# BTC Normalized (Faded Orange)
fig.add_trace(go.Scatter(x=df.index, y=df['btc_cum'], mode='lines', name='BTC-USD Normalized (Raw)', 
                         line=dict(color='rgba(247, 147, 26, 0.5)', dash='dash'), opacity=0.8), row=1, col=1) 
# ETH Normalized (Faded Grey)
fig.add_trace(go.Scatter(x=df.index, y=df['eth_cum'], mode='lines', name='ETH-USD Normalized (Raw)', 
                         line=dict(color='rgba(130, 130, 130, 0.5)', dash='dash'), opacity=0.8), row=1, col=1) 

# 2. Combined Index (The primary line)
fig.add_trace(go.Scatter(x=df.index, y=df['index_cum_smooth'], mode='lines', 
                         name=f'Combined Index EMA {index_ema_span}', line=dict(color='#0077c9', width=2)), row=1, col=1)

# 3. EMAs (on combined index)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_short'], mode='lines', name=f'EMA {ema_short}', line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_long'], mode='lines', name=f'EMA {ema_long}', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_very_long'], mode='lines', name=f'EMA {ema_very_long}', line=dict(color='purple', dash='dot')), row=1, col=1) # New EMA 72

fig.add_hline(y=1.0, line=dict(color='gray', dash='dash'), row=1, col=1) # Baseline for Normalized Price
fig.update_yaxes(title="Normalized Index (Base 1.0)", row=1, col=1)


# 4. Signals Markers
buys = df[df['signal'] == 1]
sells = df[df['signal'] == -1]
if not buys.empty:
    fig.add_trace(go.Scatter(x=buys.index, y=buys['index_cum_smooth'], mode='markers', marker_symbol='triangle-up',
                             marker_color='green', marker_size=12, name='BUY', marker_line_width=1), row=1, col=1)
if not sells.empty:
    fig.add_trace(go.Scatter(x=sells.index, y=sells['index_cum_smooth'], mode='markers', marker_symbol='triangle-down',
                             marker_color='red', marker_size=12, name='SELL', marker_line_width=1), row=1, col=1)

# 5. RSI Subplot
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name=f'RSI({rsi_length}) (EMA-Smoothed)', line=dict(color='black')), row=2, col=1)
fig.add_hrect(y0=71, y1=100, fillcolor="red", opacity=0.1, line_width=0, row=2, col=1) # Overbought Zone
fig.add_hrect(y0=0, y1=29, fillcolor="green", opacity=0.1, line_width=0, row=2, col=1) # Oversold Zone
fig.add_hline(y=50, line=dict(color='grey', dash='dot'), row=2, col=1)
fig.update_yaxes(range=[0, 100], title="RSI", row=2, col=1) 

fig.update_layout(title="BTC/ETH Combined Index Momentum Dashboard",
                  xaxis=dict(rangeslider=dict(visible=False)), height=800, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Show table of signals and diagnostics
# -----------------------
st.markdown("### Signals and Diagnostics")
if df['signal'].abs().sum() == 0:
    st.info("No signals found in the selected period with current parameters. Try adjusting settings.")
else:
    sig_df = df[df['signal'] != 0][['index_cum_smooth','EMA_short','EMA_long','EMA_very_long','RSI','volume', 'Volume_MA','signal_reason','signal']].copy()
    sig_df.index = sig_df.index.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(sig_df.tail(50))

# small metrics
st.markdown("### Summary")
st.write(f"Total cycles detected: **{len(cycles)}**")
st.write(f"Total signals detected: **{int(df['signal'].abs().sum())}** (Filtered by Triple EMA Stack)")
st.write(f"Last signal timestamp recorded: **{st.session_state.last_signal_time.strftime('%Y-%m-%d %H:%M:%S')}** (Used to prevent duplicate alerts.)")

# -----------------------
# Auto-Refresh / Manual Refresh
# -----------------------
st.markdown("---")
col_button, col_timer = st.columns([1, 4])

# Refresh button
if col_button.button(f"🔄 Refresh / Re-fetch Index Data"):
    fetch_and_process_data.clear()
    st.experimental_rerun()

# Auto-refresh timer logic
placeholder = col_timer.empty()
refresh = 300 # Fixed refresh rate for simplicity and performance
if refresh > 0:
    for i in range(refresh, 0, -1):
        with placeholder.container():
            st.markdown(f"Next auto-refresh in **{i}** seconds...")
        time.sleep(1)
    
    fetch_and_process_data.clear()
    st.experimental_rerun()
else:
    placeholder.markdown("Auto refresh is **disabled**.")