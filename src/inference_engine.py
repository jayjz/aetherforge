"""
AetherForge Hardware Engine (The Muscle)
========================================
Manages physical execution of LLM generation via llama.cpp.
Supports dynamic 'Fast-Swap' VRAM allocation based on Agent strategy.
"""

import os
import time
import gc
from typing import Dict, Any
from llama_cpp import Llama

class AetherEngine:
    # Pre-defined memory profiles for an 8GB VRAM budget (RTX 4060)
    STRATEGY_MAP = {
        "high_fidelity": 15,    # Maximizes VRAM for reasoning/coding
        "balanced": 10,         # Standard operating mode
        "aggressive_quant": 2   # Drops to System RAM for simple tasks, freeing VRAM
    }

    def __init__(self, model_path: str, vram_budget_mb: float = 8000):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model_path = model_path
        self.vram_budget_mb = vram_budget_mb
        self.current_strategy = "balanced"
        self.llm = None
        
        print(f"[Engine] Booting AetherForge Inference Core...")
        print(f"[Engine] Target Model: {os.path.basename(model_path)}")
        
        # Initial boot
        self._load_model(self.STRATEGY_MAP[self.current_strategy])
        print("[Engine] CUDA Engine Online.")

    def _load_model(self, n_gpu_layers: int):
        """Safely allocates the model into hardware memory."""
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            verbose=False
        )

    def apply_strategy(self, mode: str) -> bool:
        """
        Executes the Fast-Swap protocol. 
        Tears down the active context and reallocates VRAM boundaries.
        """
        if mode not in self.STRATEGY_MAP:
            print(f"[Engine] Invalid strategy '{mode}'. Defaulting to balanced.")
            mode = "balanced"
            
        if mode == self.current_strategy:
            return True # No-op, already in correct state
            
        target_layers = self.STRATEGY_MAP[mode]
        print(f"\n[Engine] HARDWARE OVERRIDE INITIATED.")
        print(f" -> Shifting from '{self.current_strategy}' to '{mode}'.")
        print(f" -> Reallocating VRAM for {target_layers} layers...")
        
        start_time = time.time()
        
        # 1. Destroy the current pointer
        del self.llm
        
        # 2. Force Python garbage collection to flush the GPU buffer
        gc.collect()
        
        # 3. Re-instantiate with new VRAM boundaries
        self._load_model(target_layers)
        
        swap_time = time.time() - start_time
        self.current_strategy = mode
        
        print(f"[Engine] VRAM Reallocation Complete in {swap_time:.2f}s.")
        return True

    def generate(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Executes inference using the currently active memory profile."""
        print(f"\n[Engine] Commencing generation (Mode: {self.current_strategy.upper()})...")
        
        start_time = time.time()
        
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            echo=False
        )
        
        generation_time = time.time() - start_time
        tokens_generated = output['usage']['completion_tokens']
        tps = tokens_generated / generation_time if generation_time > 0 else 0
        
        print(f" -> Time taken: {generation_time:.2f}s")
        print(f" -> Speed: {tps:.2f} tokens/sec")
        
        return {
            "text": output['choices'][0]['text'],
            "metrics": {
                "tokens_generated": tokens_generated,
                "time_seconds": generation_time,
                "tokens_per_second": tps,
                "active_strategy": self.current_strategy
            }
        }


# --- EXECUTION / TESTING ---
if __name__ == "__main__":
    target_model = r"models\DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
    
    if os.path.exists(target_model):
        engine = AetherEngine(model_path=target_model)
        
        # 1. Generate in standard balanced mode
        engine.generate("Write a one sentence summary of machine learning.", max_tokens=30)
        
        # 2. Simulate the Agent injecting a new strategy
        engine.apply_strategy("aggressive_quant")
        
        # 3. Generate under the new low-VRAM constraints
        engine.generate("What is 2+2?", max_tokens=10)
        
        # 4. Agent needs heavy coding power again
        engine.apply_strategy("high_fidelity")
        
        engine.generate("Write a python function to fetch an API.", max_tokens=50)
    else:
        print("[!] Download a model first to test hardware execution.")