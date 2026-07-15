"""
AetherForge VRAM Boundary Mapper
================================
Brute-forces the GPU memory limits to find exactly when the 
KV-Cache collides with the Model Weights on an 8GB card.
"""

import time
import gc
import sys
from llama_cpp import Llama

# Exact path to your DeepSeek model
MODEL_PATH = r"models\DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"

# We lock layers at 15 (High Fidelity) and scale the context window
TEST_LAYERS = 15
CONTEXT_SIZES = [1024, 2048, 4096, 6144, 8192, 12288]

def run_stress_test():
    print(f"=== AetherForge KV-Cache Boundary Mapping ===")
    print(f"GPU Target: RTX 4060 (8GB VRAM)")
    print(f"Weight Allocation: {TEST_LAYERS} Layers (High Fidelity)")
    print("WARNING: This script is DESIGNED to cause a hardware OOM crash.\n")

    last_successful_ctx = 0

    for ctx in CONTEXT_SIZES:
        print(f"[-] Attempting Context Window: {ctx} tokens...")
        try:
            start_time = time.time()
            # Booting the model allocates the static KV-Cache block in VRAM
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=TEST_LAYERS,
                n_ctx=ctx,
                verbose=False
            )
            boot_time = time.time() - start_time
            print(f"    [SUCCESS] VRAM block reserved in {boot_time:.2f}s.")
            
            # Execute a micro-prompt to force the CUDA cores to touch the memory
            print("    [+] Touching memory block...")
            llm("Test prompt", max_tokens=5, echo=False)
            print(f"    [SUCCESS] Inference stable at {ctx} tokens.\n")
            
            last_successful_ctx = ctx
            
            # Brutal teardown to clear the GPU for the next massive allocation
            del llm
            gc.collect()
            time.sleep(1) # Give the OS a second to reclaim the pointers
            
        except Exception as e:
            print(f"\n[FATAL CRASH] C++ Exception Caught at {ctx} tokens.")
            print(f"Error: {e}")
            break

    print(f"\n=== TEST CONCLUDED ===")
    print(f"Maximum Safe Context at {TEST_LAYERS} Layers: {last_successful_ctx} tokens.")
    print("If your terminal just violently closed without printing this summary,")
    print("the C++ CUDA backend triggered a hard abort (OOM).")

if __name__ == "__main__":
    run_stress_test()