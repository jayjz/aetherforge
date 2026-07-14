import time
import gc
from llama_cpp import Llama

# The exact path to your DeepSeek model
MODEL_PATH = r"models\DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"

def measure_fast_swap():
    print("=== AetherForge Hardware Fast-Swap Test ===")
    
    # ---------------------------------------------------------
    # PHASE 1: AGGRESSIVE QUANT (Low VRAM, mostly system RAM)
    # ---------------------------------------------------------
    print("\n[Phase 1] Booting AGGRESSIVE_QUANT Strategy (n_gpu_layers=2)...")
    start_time = time.time()
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=2,   # Bare minimum in VRAM
        n_ctx=2048,
        verbose=False
    )
    
    boot_time_1 = time.time() - start_time
    print(f" -> Phase 1 Boot Time: {boot_time_1:.2f} seconds")
    
    # Run a quick dummy prompt to warm the CUDA cores
    print(" -> Warming up cores...")
    llm("def add(a, b):", max_tokens=10, echo=False)
    
    # ---------------------------------------------------------
    # THE SWAP: Clear the VRAM instantly
    # ---------------------------------------------------------
    print("\n[Bridge] Agent requested HIGH_FIDELITY. Executing Fast-Swap...")
    swap_start = time.time()
    
    del llm           # Destroy the Python pointer
    gc.collect()      # Force Python to release the C++ memory map
    
    # ---------------------------------------------------------
    # PHASE 2: HIGH FIDELITY (Push everything into VRAM)
    # ---------------------------------------------------------
    print("[Phase 2] Booting HIGH_FIDELITY Strategy (n_gpu_layers=15)...")
    
    llm_heavy = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=15,  # Pushing the RTX 4060 hard
        n_ctx=2048,
        verbose=False
    )
    
    swap_time = time.time() - swap_start
    print(f" -> Phase 2 Swap Time: {swap_time:.2f} seconds")
    
    print("\n=== Test Complete ===")
    print(f"Total Latency to shift VRAM architecture mid-session: {swap_time:.2f} seconds")

if __name__ == "__main__":
    measure_fast_swap()