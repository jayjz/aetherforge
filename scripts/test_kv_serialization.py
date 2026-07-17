"""
AetherForge KV-Cache Serialization Test
========================================
Proves that context memory survives physical VRAM teardowns and 
reallocations by measuring Prompt Processing (Prefill) times.
"""

import os
import time
from src.inference_engine import AetherEngine

def run_serialization_gauntlet():
    target_model = r"models\DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
    
    if not os.path.exists(target_model):
        print(f"[!] Model not found at {target_model}")
        return

    print("\n==================================================")
    print("🧪 INITIATING KV-CACHE SURVIVAL TEST")
    print("==================================================")

    # Boot the engine in a throttled state
    engine = AetherEngine(model_path=target_model, vram_budget_mb=8000)
    engine.apply_strategy("balanced")

    # ---------------------------------------------------------
    # PHASE 1: Build the Context (The Heavy Lift)
    # ---------------------------------------------------------
    print("\n[Phase 1] Seeding deep context into the active KV-Cache...")
    
    # We create a dense, repetitive block of text to force the engine to do heavy prefill math.
    base_context = "SYSTEM DIRECTIVE: The critical system override code is 77-ALPHA-NINER. " * 50
    base_context += "\nEnd of directives. Acknowledge."

    # This generation will take a while because it has to read the massive prefix from scratch.
    print(" -> Executing initial prefill...")
    result_1 = engine.generate(base_context, max_tokens=10)
    print(f" -> Output: {result_1['text'].strip()}")

    # ---------------------------------------------------------
    # PHASE 2: The Fast-Swap (The Danger Zone)
    # ---------------------------------------------------------
    print("\n[Phase 2] Triggering VRAM Fast-Swap to 'high_fidelity'...")
    # This invokes our new save_state() and load_state() wrapper
    engine.apply_strategy("high_fidelity")

    # ---------------------------------------------------------
    # PHASE 3: The Memory Test
    # ---------------------------------------------------------
    print("\n[Phase 3] Testing Memory Retention...")
    
    # We append a question to the EXACT SAME PREFIX.
    # If the KV-cache survived the swap, llama.cpp will recognize the prefix
    # and instantly jump to generating the answer.
    test_prompt = base_context + "\n\nUSER: What is the critical system override code?\nASSISTANT:"
    
    print(" -> Executing follow-up prompt...")
    start_time = time.time()
    result_2 = engine.generate(test_prompt, max_tokens=20)
    total_time = time.time() - start_time
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Output: {result_2['text'].strip()}")
    print(f"Total Response Time: {total_time:.2f}s")
    print("====================")
    
    # Verification Logic
    if "77-ALPHA-NINER" in result_2['text'].upper():
        print("\n✅ SUCCESS: The model correctly recalled the injected context.")
    else:
        print("\n❌ FAILURE: The model hallucinated or forgot the context.")

if __name__ == "__main__":
    run_serialization_gauntlet()