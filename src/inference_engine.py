"""
AetherForge Hardware Engine (The Muscle)
========================================
Manages physical execution of LLM generation via llama.cpp.
Supports dynamic 'Fast-Swap' VRAM allocation and KV-Cache preservation.
"""

import os
import time
import gc
from typing import Dict, Any
from llama_cpp import Llama
from src.config import settings

class AetherEngine:
    def __init__(self, model_path: str, vram_budget_mb: float = 8000, n_ctx: int = 4096):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model_path = model_path
        self.vram_budget_mb = vram_budget_mb
        self.n_ctx = n_ctx
        self.current_strategy = "balanced"
        self.llm = None
        
        # Build memory profile dynamically from centralized config (No more magic numbers)
        self.strategy_map = {
            "high_fidelity": settings.layers_high_fidelity,
            "balanced": settings.layers_balanced,
            "aggressive_quant": settings.layers_aggressive_quant
        }
        
        print(f"[Engine] Booting AetherForge Inference Core...")
        print(f"[Engine] Target Model: {os.path.basename(model_path)}")
        print(f"[Engine] Strategy Layer Scheme: {self.strategy_map}")
        
        # Initial boot
        self._load_model(self.strategy_map[self.current_strategy])
        print("[Engine] CUDA Engine Online.")

    def _load_model(self, n_gpu_layers: int):
        """Safely allocates the model into hardware memory boundaries."""
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False
        )
        
    def count_tokens(self, text: str) -> int:
        """Uses the active native model tokenizer to calculate exact context depth."""
        if not self.llm:
            print("[Engine] WARNING: Token count requested while LLM instance was offline.")
            return 0
        tokens = self.llm.tokenize(text.encode('utf-8'))
        return len(tokens)

    def apply_strategy(self, mode: str) -> bool:
        """
        Executes the Fast-Swap protocol with State Serialization. 
        Extracts KV-Cache, reallocates VRAM boundaries, and restores memory.
        """
        if mode not in self.strategy_map:
            print(f"[Engine] Invalid strategy '{mode}'. Defaulting to balanced.")
            mode = "balanced"
            
        if mode == self.current_strategy:
            return True # No-op, already in correct state
            
        target_layers = self.strategy_map[mode]
        print(f"\n[Engine] HARDWARE OVERRIDE INITIATED.")
        print(f" -> Shifting from '{self.current_strategy}' to '{mode}'.")
        print(f" -> Reallocating VRAM for {target_layers} layers...")
        
        start_time = time.time()
        
        # 1. KV-Cache Extraction
        saved_state = None
        if self.llm:
            try:
                print("[Engine] Serializing active KV-Cache to System RAM...")
                state_start = time.time()
                saved_state = self.llm.save_state()
                print(f" -> Extraction complete in {time.time() - state_start:.2f}s")
            except Exception as e:
                print(f"[!] WARNING: State extraction failed: {e}. Proceeding with cold restart.")
        
        # 2. Hardware Teardown
        del self.llm
        gc.collect()
        
        # 3. Hardware Rebuild
        self._load_model(target_layers)
        
        # 4. KV-Cache Injection
        if saved_state:
            try:
                print("[Engine] Injecting KV-Cache into new VRAM layout...")
                restore_start = time.time()
                self.llm.load_state(saved_state)
                print(f" -> Injection complete in {time.time() - restore_start:.2f}s")
            except Exception as e:
                print(f"[CRITICAL] KV-Cache restoration failed: {e}. Memory wiped.")
        
        swap_time = time.time() - start_time
        self.current_strategy = mode
        
        print(f"[Engine] Fast-Swap Protocol Complete in {swap_time:.2f}s.")
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