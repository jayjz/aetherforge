"""
AetherForge Reference Client: The Resilient Agent
=================================================
Demonstrates Agent-Infrastructure Negotiation.
This agent autonomously handles L7 JSON contracts from the AetherForge 
control plane, including thermal backoffs and context ceiling truncations.
"""

import requests
import time
import sys

API_BASE = "http://127.0.0.1:8000"

def agent_loop():
    print("\n" + "="*50)
    print("🤖 RESILIENT AGENT INITIALIZED")
    print("="*50)

    # 1. STRATEGY DISCOVERY
    try:
        print("\n-> [1] Discovering Hypervisor Capabilites...")
        metrics = requests.get(f"{API_BASE}/system/metrics").json()
        print(f"   [+] Active Strategy: {metrics['active_strategy']}")
        print(f"   [+] Hardware State: {'SIMULATED' if metrics['engine_state'] == 'simulation' else 'ONLINE'}")
        
        if metrics.get("thermal_lock_active"):
            print("   [!] WARNING: System currently thermally locked.")
    except requests.exceptions.ConnectionError:
        print("\n[FATAL] Cannot connect to Control Plane. Is Uvicorn running?")
        sys.exit(1)

    # 2. STRATEGY NEGOTIATION
    print("\n-> [2] Negotiating VRAM Strategy...")
    strat_payload = {
        "mode": "high_fidelity",
        "estimated_context_tokens": 1000,
        "expected_output_tokens": 100
    }
    strat_resp = requests.post(f"{API_BASE}/system/strategy", json=strat_payload)
    
    if strat_resp.status_code == 200:
        data = strat_resp.json()
        if data.get("status") == "rejected":
            print(f"   [-] Gatekeeper Rejected Swap: {data.get('error')} ({data.get('active_mode')} maintained)")
        else:
            print(f"   [+] Strategy Applied: {data.get('active_mode')}")
    elif strat_resp.status_code == 503:
        err = strat_resp.json()["detail"]
        print(f"   [!] 503 {err['error'].upper()}: Sleeping {err['retry_after_seconds']}s...")
        time.sleep(err["retry_after_seconds"])

    # 3. GENERATION & CIRCUIT BREAKER SURVIVAL
    print("\n-> [3] Commencing Heavy Generation Task (Triggering 413/503 limits)...")
    
    # We simulate an agent making a mistake: sending a 50,000 character prompt
    prompt_content = "SYSTEM_MEMORY_DUMP " * 2500 
    
    while True:
        gen_payload = {
            "prompt": prompt_content,
            "max_tokens": 50,
            "strategy": {"mode": "balanced"}
        }
        
        print(f"   [>] Submitting payload... ({len(prompt_content)} chars)")
        gen_resp = requests.post(f"{API_BASE}/generate", json=gen_payload)
        
        # SUCCESS
        if gen_resp.status_code == 200:
            data = gen_resp.json()
            tps = data.get("metrics", {}).get("tokens_per_second", 0)
            tokens = data.get("metrics", {}).get("tokens_generated", 0)
            print(f"   [✅] SUCCESS: Generated {tokens} tokens @ {tps:.2f} TPS")
            break
            
        # THERMAL OR QUEUE LOCK (503)
        elif gen_resp.status_code == 503:
            err = gen_resp.json().get("detail", {})
            wait_time = err.get("retry_after_seconds", 5)
            print(f"   [⏳] 503 {err.get('error', 'BUSY').upper()}: Control plane rejected traffic. Backing off for {wait_time}s...")
            time.sleep(wait_time)
            
        # CONTEXT CEILING BREACH (413)
        elif gen_resp.status_code == 413:
            err = gen_resp.json().get("detail", {})
            max_allowed = err.get("max_allowed", 4000)
            attempted = err.get("attempted", "Unknown")
            action = err.get("action", "truncate")
            
            print(f"   [✂️] 413 CONTEXT CEILING HIT: Attempted {attempted} tokens. Hardware max is {max_allowed}.")
            print(f"   [✂️] Executing recommended API action: '{action}'")
            
            # Agent autonomously complies with infrastructure limits
            # (Rough conversion: 1 token ~= 4 chars)
            safe_char_limit = int(max_allowed * 3.5)
            prompt_content = prompt_content[:safe_char_limit]
            
        else:
            print(f"   [❌] UNHANDLED EXCEPTION {gen_resp.status_code}: {gen_resp.text}")
            break

if __name__ == "__main__":
    agent_loop()