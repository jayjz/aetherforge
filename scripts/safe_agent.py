"""
AetherForge Safe Agent Client
=============================
Demonstrates an external AI agent interacting with the AetherForge API.
The agent discovers the VRAM optimization tool, attempts a strategy swap 
based on workload, monitors hardware health, and handles 503 thermal locks.
"""

import time
import requests
import json
import os

API_BASE = os.getenv("AETHER_API_BASE", "http://127.0.0.1:8000")

def print_header(title: str):
    print(f"\n{'='*50}\n[AGENT] {title}\n{'='*50}")

def agent_loop():
    print_header("BOOT SEQUENCE & TOOL DISCOVERY")
    
    try:
        print("-> Querying /system/tools for hypervisor capabilities...")
        tool_response = requests.get(f"{API_BASE}/system/tools")
        tool_schema = tool_response.json()
        print(f"-> Discovered capability: {tool_schema['function']['name']}")
    except requests.exceptions.ConnectionError:
        print("FATAL: AetherForge Control Plane is offline. Exiting.")
        return

    print_header("PRE-FLIGHT HARDWARE CHECK")
    metrics = requests.get(f"{API_BASE}/system/metrics").json()
    print(f"Engine State: {metrics['engine_state'].upper()} | Thermal Lock: {'ACTIVE' if metrics['thermal_lock_active'] else 'CLEAR'}")
    
    # 503 Resilience Loop
    while metrics["thermal_lock_active"]:
        print("-> WATCHDOG ACTIVE: GPU is cooling. Agent enters 5s standby mode...")
        time.sleep(5)
        metrics = requests.get(f"{API_BASE}/system/metrics").json()
    print("-> Hardware cleared for execution.")

    print_header("STRATEGY NEGOTIATION (TOOL USAGE)")
    print("-> Agent Task: Generate a massive 2000-token structural report.")
    print("-> Agent Action: Requesting 'high_fidelity' VRAM allocation via tool schema.")
    
    strategy_payload = {
        "mode": "high_fidelity",
        "estimated_context_tokens": 150,
        "expected_output_tokens": 2000
    }
    
    strat_response = requests.post(f"{API_BASE}/system/strategy", json=strategy_payload)
    if strat_response.status_code == 200:
        strat_data = strat_response.json()
        if strat_data["status"] == "strategy_applied":
            print(f"-> Gatekeeper ACCEPTED swap. Active mode is now: {strat_data['active_mode']}")
        else:
            print(f"-> Gatekeeper REJECTED swap: {strat_data.get('reason')}. Proceeding with: {strat_data['active_mode']}")
    elif strat_response.status_code == 503:
        print("-> 503 REJECTED: Thermal limit breached during negotiation. Aborting.")
        return

    print_header("EXECUTING WORKLOAD")
    gen_payload = {
        "prompt": "Analyze the structural integrity of the bridge.",
        "max_tokens": 2000
    }
    
    print(f"-> Submitting generation payload...")
    gen_response = requests.post(f"{API_BASE}/generate", json=gen_payload)
    
    if gen_response.status_code == 200:
        data = gen_response.json()
        print(f"-> SUCCESS: Generated {data['metrics']['tokens_generated']} tokens at {data['metrics']['tokens_per_second']:.2f} TPS.")
    elif gen_response.status_code == 413:
        print(f"-> 413 REJECTED: Context violates hardware safety limits.")
        print("-> Agent Recovery: Would summarize context and retry.")
    else:
        print(f"-> ERROR {gen_response.status_code}: {gen_response.text}")

if __name__ == "__main__":
    agent_loop()