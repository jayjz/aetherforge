"""
AetherForge Operator Lens (Thin UI)
===================================
A lightweight Streamlit dashboard that visualizes the L7 control plane
metrics and the durable JSON audit trail.
"""

import streamlit as st
import requests
import pandas as pd
import json
import os
import time

st.set_page_config(page_title="AetherForge Ops", page_icon="🛡️", layout="wide")
API_URL = os.getenv("AETHER_API_BASE", "http://127.0.0.1:8000")

def fetch_metrics():
    try:
        res = requests.get(f"{API_URL}/system/metrics", timeout=2)
        if res.status_code == 200:
            return res.json()
    except Exception:
        return None

def fetch_audit_logs(filepath="logs/audit_trail.jsonl", n=15):
    if not os.path.exists(filepath):
        return []
    logs = []
    try:
        with open(filepath, "r") as f:
            for line in f.readlines()[-n:]:
                if line.strip():
                    logs.append(json.loads(line.strip()))
    except Exception:
        pass
    return reversed(logs) # Newest first

# --- UI LAYOUT ---
st.title("🛡️ AetherForge Control Plane")
st.markdown("Real-time telemetry and Agent-Infrastructure Negotiation Audit Trail.")

st.sidebar.header("Controls")
auto_refresh = st.sidebar.checkbox("Live Polling (2s)", value=True)

metrics = fetch_metrics()
if not metrics:
    st.error(f"Cannot connect to API at {API_URL}. Is Uvicorn running?")
    st.stop()

# --- KPI ROW ---
col1, col2, col3, col4 = st.columns(4)
is_locked = metrics.get('thermal_lock_active', False)

col1.metric("Engine State", metrics.get('engine_state', 'UNKNOWN').upper())
col2.metric("Active Strategy", metrics.get('active_strategy', 'N/A').upper())
col3.metric("Thermal Lock (503)", "🚨 ACTIVE" if is_locked else "✅ CLEAR")

vram_pct = metrics.get('vram_pressure', {}).get('utilization_pct', 0)
temp_c = metrics.get('silicon_vitals', {}).get('temp_c', 0)
col4.metric("VRAM / Temp", f"{vram_pct:.1f}% / {temp_c}°C")

st.divider()

# --- DETAILS ROW ---
col_tps, col_logs = st.columns([1, 2])

with col_tps:
    st.subheader("Gatekeeper ROI Profiles")
    st.markdown("Exponential Moving Average (EMA) of TPS by strategy.")
    profiles = metrics.get('performance_baselines', {})
    if profiles:
        df_tps = pd.DataFrame([{"Strategy": k, "Live TPS": round(v["live_tps"], 2)} for k, v in profiles.items()])
        st.dataframe(df_tps, hide_index=True, use_container_width=True)

with col_logs:
    st.subheader("Durable Audit Trail (JSONL)")
    logs = list(fetch_audit_logs())
    if logs:
        # Flatten the JSON for clean dataframe display
        flat_logs = []
        for lg in logs:
            row = {"Time": lg.get("timestamp", "").split(" ")[1] if lg.get("timestamp") else "", "Event": lg.get("event", "")}
            details = lg.get("details", {})
            if details:
                row["Reason"] = details.get("reason", "")
                row["Target"] = details.get("target_mode", "")
                row["Ctx Size"] = details.get("context_tokens", "")
            flat_logs.append(row)
            
        st.dataframe(pd.DataFrame(flat_logs), hide_index=True, use_container_width=True)
    else:
        st.info("No audit events logged yet. Fire the Resilient Agent to generate events.")

if auto_refresh:
    time.sleep(2)
    st.rerun()