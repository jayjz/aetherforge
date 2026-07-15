"""
AetherForge Self-Regulating Agent Loop
======================================
DeepSeek acts as its own orchestrator. It evaluates the prompt, 
chooses a physical VRAM strategy, triggers the FastAPI swap, 
and then executes the prompt.
"""

import requests
import json
import re
import time

API_URL = "http://127.0.0.1:8000"

def chat_with_aether(prompt: str, max_tokens: int = 200):
    """Sends a raw prompt to the AetherForge API."""
    try:
        response = requests.post(
            f"{API_URL}/generate", 
            json={"prompt": prompt, "max_tokens": max_tokens, "simulate": False}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[!] API Error: {e}")
        return None

def trigger_hardware_swap(mode: str):
    """Hits the Control Plane to execute the 5.8s VRAM swap."""
    try:
        response = requests.post(
            f"{API_URL}/system/strategy", 
            json={"mode": mode}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[!] Hardware Swap Error: {e}")
        return None

def run_agent_task(user_task: str):
    print(f"\n==================================================")
    print(f"🎯 NEW TASK: {user_task}")
    print(f"==================================================")
    
    # ---------------------------------------------------------
    # PHASE 1: The Brain (Strategy Evaluation)
    # ---------------------------------------------------------
    print("\n[Agent] Phase 1: Evaluating hardware requirements...")
    
    eval_prompt = f"""You are Aether, an autonomous AI managing your own physical GPU memory.
You have three VRAM strategies available:
- "high_fidelity": Use for complex coding, math, and deep reasoning.
- "balanced": Use for standard interactions.
- "aggressive_quant": Use for simple trivia, basic text routing, and summarization.

USER TASK: "{user_task}"

Analyze the task and choose the best strategy. 
Respond ONLY with a valid JSON object. Do not include markdown blocks or any other text.
FORMAT: {{"mode": "chosen_mode"}}

Response:"""

    eval_result = chat_with_aether(eval_prompt, max_tokens=50)
    if not eval_result: return
    
    raw_output = eval_result["text"].strip()
    print(f" -> Raw LLM Output: {raw_output}")
    
    # Use Regex to hunt for the JSON payload in case the LLM hallucinates extra text
    match = re.search(r'\{\s*"mode"\s*:\s*"([^"]+)"\s*\}', raw_output)
    
    if match:
        chosen_mode = match.group(1).lower()
        if chosen_mode not in ["high_fidelity", "balanced", "aggressive_quant"]:
            chosen_mode = "balanced" # Failsafe
    else:
        print("[!] Failed to parse JSON. Defaulting to balanced.")
        chosen_mode = "balanced"

    print(f" -> Decision: Lock GPU into [{chosen_mode.upper()}] mode.")

    # ---------------------------------------------------------
    # PHASE 2: The Muscle (Hardware Swap)
    # ---------------------------------------------------------
    print(f"\n[Agent] Phase 2: Commanding API to restructure VRAM...")
    swap_result = trigger_hardware_swap(chosen_mode)
    
    if swap_result and swap_result.get("status") == "hardware_strategy_applied":
        print(" -> Hardware swap confirmed by Hypervisor.")
    else:
        print(" -> Strategy unchanged (already in optimal state).")

    # ---------------------------------------------------------
    # PHASE 3: Execution (Running the Task)
    # ---------------------------------------------------------
    print(f"\n[Agent] Phase 3: Executing task under new hardware constraints...")
    
    execution_prompt = f"User Task: {user_task}\nAnswer:"
    final_result = chat_with_aether(execution_prompt, max_tokens=300)
    
    if final_result:
        print("\n=== FINAL OUTPUT ===")
        print(final_result["text"].strip())
        print("====================")
        metrics = final_result.get("metrics", {})
        print(f"⏱️ Speed: {metrics.get('tokens_per_second', 0):.2f} t/s | Strategy: {metrics.get('active_strategy', 'unknown')}")

# --- THE GAUNTLET ---
if __name__ == "__main__":
    # Task 1: A simple trivia question. The Agent should drop VRAM to save resources.
    run_agent_task("What is the capital of France? Give me a one sentence answer.")
    
    time.sleep(2)
    
    # Task 2: A complex coding request. The Agent should demand maximum VRAM.
    run_agent_task("Write a Python function to implement the A* pathfinding algorithm on a 2D grid.")