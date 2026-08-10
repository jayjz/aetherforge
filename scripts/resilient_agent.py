"""
AetherForge Resilient Agent Client
==================================
Demonstrates production-grade resilience. The agent handles 413 Context Ceilings 
by autonomously chunking memory, and 503 Thermal/Queue Locks via exponential backoff.
"""

import time
import requests
import json
import os

API_BASE = os.getenv("AETHER_API_BASE", "http://127.0.0.1:8000")

def print_header(title: str):
    print(f"\n{'='*50}\n[AGENT] {title}\n{'='*50}")

def resilient_post(endpoint: str, payload: dict, max_retries=5):
    """Executes a request with built-in recovery for AetherForge constraints."""
    base_wait = 2
    
    for attempt in range(max_retries):
        response = requests.post(f"{API_BASE}{endpoint}", json=payload)
        
        if response.status_code == 200:
            return response.json()
            
        elif response.status_code == 503:
            # Server is alive, but GPU/Queue is locked
            error_data = response.json()
            retry_after = int(response.headers.get("Retry-After", base_wait))
            
            print(f" -> [503] {error_data['detail']}")
            print(f" -> [Agent] Backing off for {retry_after}s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(retry_after)
            
            # If it's a thermal lock, actively poll the vitals until cooled
            if "SYSTEM LOCKED" in error_data['detail']:
                metrics = requests.get(f"{API_BASE}/system/metrics").json()
                while metrics.get("thermal_lock_active", False):
                    print(" -> [Watchdog] GPU still cooling. Polling in 5s...")
                    time.sleep(5)
                    metrics = requests.get(f"{API_BASE}/system/metrics").json()
                print(" -> [Agent] Hardware normalized. Resuming.")
                
        elif response.status_code == 413:
            # Prompt is too large for the physical VRAM limits
            print(f" -> [413] Payload too large. Context limit breached.")
            print(" -> [Agent] Autonomously truncating/summarizing context to fit hardware bounds...")
            if "prompt" in payload:
                # Naive chunking for demonstration: cut prompt size in half
                current_len = len(payload["prompt"])
                payload["prompt"] = payload["prompt"][:current_len // 2] + "\n[SYSTEM: CONTEXT TRUNCATED DUE TO MEMORY LIMITS]"
            else:
                return {"error": "Cannot chunk non-prompt payload."}
                
        else:
            print(f" -> [FATAL] Unhandled error {response.status_code}: {response.text}")
            return None
            
    print(" -> [FATAL] Max retries exhausted.")
    return None

def run_resilience_demo():
    print_header("SCENARIO 1: THE CONTEXT BOMB (413 RECOVERY)")
    # We send a massive prompt (AetherForge defaults to ~8192 tokens max)
    massive_prompt = "Agent memory log... " * 10000 
    gen_payload = {
        "prompt": massive_prompt,
        "max_tokens": 100
    }
    
    print("-> Attempting to process massive memory array...")
    result = resilient_post("/generate", gen_payload)
    if result:
        print(f"✅ SUCCESS: Survived context limitation. Tokens Generated: {result['metrics']['tokens_generated']}")

if __name__ == "__main__":
    run_resilience_demo()