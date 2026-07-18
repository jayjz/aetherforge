"""
AetherForge KV-Cache Stress Tester
==================================
Empirically tests the limits of Fast-Swap KV serialization.
Tracks RAM bloat, extraction latencies, and hard C++ limits.
"""

import os
import sys
import time
import psutil

try:
    from src.inference_engine import AetherEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

from src.config import settings

# Context breakpoints to test
TEST_SIZES = [1024, 4096, 8192, 16384]

def get_ram_mb():
    """Returns the current system RAM footprint of the Python process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def generate_dummy_context(target_tokens: int) -> str:
    """Generates a predictable context block to fill the KV-Cache."""
    # A standard sentence is roughly 10 tokens.
    sentence = "The hypervisor must maintain absolute state coherence across physical memory boundaries. "
    multiplier = max(1, target_tokens // 10)
    return sentence * multiplier

def run_stress_test():
    if not HAS_ENGINE:
        print("❌ [ABORT] CUDA Engine not detected. Run this on the Windows hardware.")
        sys.exit(1)

    print("🧪 INITIATING AETHERFORGE KV-CACHE STRESS TEST")
    print("=================================================")
    
    # Force max context window for the engine
    engine = AetherEngine(
        model_path=settings.model_path,
        vram_budget_mb=settings.vram_budget_mb,
        n_ctx=32768  # Override config to ensure deep limits are tested
    )

    results = []

    for target_size in TEST_SIZES:
        print(f"\n[Test Case] Context Target: {target_size} tokens")
        print("-" * 50)
        
        context = generate_dummy_context(target_size)
        actual_tokens = engine.count_tokens(context)
        print(f" -> Actual Token Count: {actual_tokens}")
        
        try:
            # 1. Fill the KV Cache
            print(" -> [Prefill] Pumping tokens into active VRAM...")
            t0 = time.perf_counter()
            # Force processing by asking for 1 token
            engine.generate(prompt=context + "\nOutput status:", max_tokens=1)
            prefill_time = time.perf_counter() - t0
            print(f" -> Prefill complete in {prefill_time:.2f}s")
            
            ram_before = get_ram_mb()
            
            # 2. Force the Fast-Swap
            print(" -> [Swap] Triggering physical VRAM reallocation...")
            target_mode = "high_fidelity" if engine.current_strategy == "balanced" else "balanced"
            
            swap_result = engine.apply_strategy(target_mode)
            metrics = swap_result.get("metrics", {})
            
            ram_after = get_ram_mb()
            ram_delta = ram_after - ram_before
            
            extract_time = metrics.get('extract_seconds', 0.0)
            inject_time = metrics.get('inject_seconds', 0.0)
            
            print(f" -> Swap Success! Extract: {extract_time:.2f}s | Inject: {inject_time:.2f}s")
            print(f" -> RAM Footprint Delta during swap: {ram_delta:+.2f} MB")
            
            results.append({
                "tokens": actual_tokens,
                "extract_s": extract_time,
                "inject_s": inject_time,
                "ram_delta_mb": ram_delta,
                "status": "PASS"
            })
            
        except MemoryError:
            print("\n❌ [FATAL] Python MemoryError: RAM budget exceeded.")
            results.append({"tokens": actual_tokens, "status": "OOM (RAM)"})
            break
        except Exception as e:
            # Note: A hard C++ segfault will kill the process entirely and bypass this.
            print(f"\n❌ [FATAL] Engine Exception: {e}")
            results.append({"tokens": actual_tokens, "status": "CRASH"})
            break

    # Print Summary Table
    print("\n\n📊 STRESS TEST SUMMARY")
    print("==================================================================")
    print(f"{'Tokens':<10} | {'Status':<10} | {'Extract (s)':<12} | {'Inject (s)':<12} | {'RAM Delta (MB)':<15}")
    print("-" * 66)
    for r in results:
        if r["status"] == "PASS":
            print(f"{r['tokens']:<10} | {r['status']:<10} | {r['extract_s']:<12.2f} | {r['inject_s']:<12.2f} | {r['ram_delta_mb']:<15.2f}")
        else:
            print(f"{r['tokens']:<10} | {r['status']:<10} | {'-':<12} | {'-':<12} | {'-':<15}")
    print("==================================================================")
    
    print("\n[NOTE] If this script aborted abruptly without printing the summary, you hit a hard C++ segmentation fault.")

if __name__ == "__main__":
    run_stress_test()