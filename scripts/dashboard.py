"""
AetherForge Telemetry Dashboard
===============================
A Streamlit operator view that polls the FastAPI control plane in real-time.
Visualizes VRAM pressure, Gatekeeper ROI math, and thermal circuit breakers.
"""

import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AetherForge Ops", page_icon="🛡️", layout="wide")
API_URL = os.getenv("AETHER_API_BASE", "http://127.0.0.1:8000")

# --- DATA FETCHERS ---
def fetch_metrics():
    try:
        res = requests.get(f"{API_URL}/system/metrics", timeout=2)
        if res.status_code == 200:
            return res.json()
    except Exception:
        return None
    return None

def fetch_cache():
    try:
        res = requests.get(f"{API_URL}/system/cache", timeout=2)
        if res.status_code == 200:
            return res.json()
    except Exception:
        return None
    return None

def tail_logs(filepath, n=10):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return f.readlines()[-n:]

# --- UI LAYOUT ---
st.title("🛡️ AetherForge Control Plane")
st.markdown("Real-time telemetry and Gatekeeper observability for local LLM orchestration.")

# Sidebar controls
st.sidebar.header("Operator Controls")
auto_refresh = st.sidebar.checkbox("Live Polling (2s)", value=True)

# Initialize session state for charting history
if 'history' not in st.session_state:
    st.session_state.history = {'time': [], 'temp': [], 'vram': []}

metrics = fetch_metrics()
cache = fetch_cache()

if not metrics:
    st.error(f"Cannot connect to AetherForge Control Plane at {API_URL}. Is Uvicorn running?")
    st.stop()

# Update History
current_time = datetime.fromtimestamp(metrics['timestamp']).strftime('%H:%M:%S')
st.session_state.history['time'].append(current_time)
st.session_state.history['temp'].append(metrics['silicon_vitals'].get('temp_c', 0))
st.session_state.history['vram'].append(metrics['vram_pressure']['utilization_pct'])

# Keep last 50 data points to prevent memory bloat
if len(st.session_state.history['time']) > 50:
    for key in st.session_state.history:
        st.session_state.history[key].pop(0)

# --- Top Level KPIs ---
col1, col2, col3, col4 = st.columns(4)

is_locked = metrics.get('thermal_lock_active', False)
thermal_status = "🚨 ACTIVE (503)" if is_locked else "✅ CLEAR"

col1.metric("Thermal Circuit Breaker", thermal_status, delta="-LOCKED" if is_locked else None, delta_color="inverse")
col2.metric("Active Strategy", metrics['active_strategy'].upper())
col3.metric("GPU Temp (°C)", f"{metrics['silicon_vitals'].get('temp_c', 0):.1f}°C")
col4.metric("VRAM Utilization", f"{metrics['vram_pressure']['utilization_pct']:.1f}%")

st.divider()

# --- Charts ---
col_chart, col_tps = st.columns([2, 1])

with col_chart:
    st.subheader("Silicon Vitals")
    df = pd.DataFrame({
        'Time': st.session_state.history['time'],
        'Temp (°C)': st.session_state.history['temp'],
        'VRAM (%)': st.session_state.history['vram']
    }).set_index('Time')
    st.line_chart(df, color=["#FF4B4B", "#0068C9"], height=300)

with col_tps:
    st.subheader("Gatekeeper ROI Profiles")
    st.markdown("Exponential Moving Average (EMA) of generation TPS.")
    profiles = metrics['performance_baselines']
    tps_data = [{"Strategy": mode, "Live TPS": data["live_tps"]} for mode, data in profiles.items()]
    st.dataframe(pd.DataFrame(tps_data), hide_index=True, use_container_width=True)

st.divider()

# --- Logs & Context ---
col_cache, col_logs = st.columns(2)

with col_cache:
    st.subheader("VRAM Memory Snapshot")
    if cache:
        st.code(f"Engine State: {cache['status'].upper()}\nBudget Used: {cache['current_vram_usage_mb']:.0f} / {cache['vram_budget_mb']} MB\nActive Experts in VRAM: {len(cache['active_experts_in_vram'])}", language="yaml")
    else:
        st.warning("Cache API offline.")

with col_logs:
    st.subheader("Hardware Safety Audit Trail")
    safety_logs = tail_logs("logs/hardware_safety.log", n=8)
    if safety_logs:
        st.code("".join(safety_logs), language="log")
    else:
        st.info("No safety events logged recently.")

# Polling loop
if auto_refresh:
    time.sleep(2)
    st.rerun()