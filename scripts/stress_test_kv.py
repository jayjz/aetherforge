"""
AetherForge KV-Cache Stress Tester (v2)
=======================================
Empirically tests the limits of Fast-Swap KV serialization.
Writes telemetry to disk incrementally to survive hard C++ segfaults.
"""

import os
import sys
import time
import csv
import psutil

try:
    from src.inference_engine import AetherEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

from src.config import settings

# Progressive overload array. Pushing to 32k to intentionally find the 8GB VRAM ceiling.
TEST_SIZES = [1024, 4096, 8192, 16384, 32768]
OUTPUT_FILE = "kv_stress_results.csv"

def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def generate_dummy_context(target_tokens: int) -> str:
    sentence = "The hypervisor must maintain absolute state coherence across physical memory boundaries. "
    multiplier = max(1, target_tokens // 10)
    return sentence * multiplier

def append_to_csv(record: dict):
    """Writes to disk immediately to survive process death."""
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["tokens", "status", "extract_s", "inject_s", "ram_delta_mb"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def run_stress_test():
    if not HAS_ENGINE:
        print("❌ [ABORT] CUDA Engine not detected. Run this on the Windows hardware.")
        sys.exit(1)

    print("🧪 INITIATING AETHERFORGE KV-CACHE STRESS TEST")
    print(f" -> Output will stream directly to {OUTPUT_FILE}")
    print("=================================================")
    
    engine = AetherEngine(
        model_path=settings.model_path,
        vram_budget_mb=settings.vram_budget_mb,
        n_ctx=32768
    )

    for target_size in TEST_SIZES:
        print(f"\n[Test Case] Context Target: {target_size} tokens")
        print("-" * 50)
        
        context = generate_dummy_context(target_size)
        
        # Guard against engine API drift - ensure fallback if count_tokens isn't perfect yet
        actual_tokens = getattr(engine, "count_tokens", lambda x: len(x) // 4)(context)
        print(f" -> Actual Token Count: {actual_tokens}")
        
        record = {"tokens": actual_tokens, "status": "PENDING", "extract_s": 0.0, "inject_s": 0.0, "ram_delta_mb": 0.0}
        
        try:
            # Note: Using generate() for prefill is tech-debt, but the safest way to force 
            # llama.cpp to evaluate the graph without raw ctypes bindings.
            print(" -> [Prefill] Pumping tokens into active VRAM...")
            engine.generate(prompt=context + "\nOutput status:", max_tokens=1)
            
            ram_before = get_ram_mb()
            
            print(" -> [Swap] Triggering physical VRAM reallocation...")
            target_mode = "high_fidelity" if engine.current_strategy == "balanced" else "balanced"
            
            swap_result = engine.apply_strategy(target_mode)
            
            # Handle potential API drift if apply_strategy doesn't return the dict yet
            if isinstance(swap_result, dict):
                metrics = swap_result.get("metrics", {})
            else:
                metrics = {"extract_seconds": 0.0, "inject_seconds": 0.0}
            
            ram_after = get_ram_mb()
            
            record["extract_s"] = round(metrics.get('extract_seconds', 0.0), 3)
            record["inject_s"] = round(metrics.get('inject_seconds', 0.0), 3)
            record["ram_delta_mb"] = round(ram_after - ram_before, 2)
            record["status"] = "PASS"
            
            print(f" -> Swap Success! Extract: {record['extract_s']}s | Inject: {record['inject_s']}s")
            
        except MemoryError:
            print("\n❌ [FATAL] Python MemoryError: RAM budget exceeded.")
            record["status"] = "OOM_RAM"
            append_to_csv(record)
            break
        except Exception as e:
            print(f"\n❌ [FATAL] Engine Exception: {e}")
            record["status"] = "CRASH_EXCEPTION"
            append_to_csv(record)
            break
            
        append_to_csv(record)

if __name__ == "__main__":
    run_stress_test()