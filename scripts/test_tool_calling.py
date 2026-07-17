"""
AetherForge Tool Calling Verification
======================================
Proves that a standard agent can dynamically discover the hypervisor's 
capabilities and successfully trigger a VRAM hardware swap.
"""

import requests
import json
import time

API_URL = "http://127.0.0.1:8000"

def run_agent_simulation():
    print("🤖 [Agent] Booting up. Discovering available tools...")
    
    # 1. Agent discovers the tool schema
    try:
        response = requests.get(f"{API_URL}/system/tools")
        tool_schema = response.json()
        print(f" -> Discovered Tool: {tool_schema['function']['name']}")
        print(f" -> Description: {tool_schema['function']['description'][:80]}...")
    except requests.exceptions.ConnectionError:
        print("[!] Cannot connect to AetherForge API. Is uvicorn running?")
        return

    print("\n🤖 [Agent] I need to write a complex Python API wrapper.")
    print("🤖 [Agent] Analyzing tool constraints... Selecting 'high_fidelity' mode.")
    
    # 2. Agent formulates the tool call payload based on the schema
    # In a real framework (like LangGraph), the LLM generates this JSON automatically.
    tool_call_payload = {
        "mode": "high_fidelity",
        "context_text": "User: Write a python wrapper for the Stripe API. Assistant: I will begin drafting the code.",
        "expected_output_tokens": 500
    }

    print("\n🤖 [Agent] Executing tool: aetherforge_optimize_vram")
    print(f" -> Payload: {json.dumps(tool_call_payload, indent=2)}")
    
    # 3. Agent executes the tool
    start_time = time.time()
    result = requests.post(f"{API_URL}/system/strategy", json=tool_call_payload)
    
    print("\n=== TOOL EXECUTION RESULT ===")
    print(f"Response HTTP: {result.status_code}")
    print(f"Response JSON: {json.dumps(result.json(), indent=2)}")
    print(f"Execution Time: {time.time() - start_time:.2f}s")
    
    if result.status_code == 200 and result.json().get("status") == "hardware_strategy_applied":
        print("✅ SUCCESS: Agent autonomously optimized hardware VRAM.")
    else:
        print("❌ FAILURE or REJECTION: Gatekeeper blocked the swap or simulation mode is active.")

if __name__ == "__main__":
    run_agent_simulation()