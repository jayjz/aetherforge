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

class LlamaEngine:    
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

    def apply_strategy(self, mode: str) -> dict:
        """
        Executes a Fast-Swap protocol by extracting the active state,
        tearing down the model constraints, and re-allocating VRAM layers.
        """
        metrics = {"extract_seconds": 0.0, "reload_seconds": 0.0, "inject_seconds": 0.0}
        
        if mode == self.current_strategy and self.llm is not None:
            return {"success": True, "metrics": metrics}
            
        print(f"[Engine] Commencing physical hardware shift to '{mode}'...")
        
        # 1. Measure KV Extraction / Serialization to System RAM
        if self.llm is not None:
            t_start_extract = time.perf_counter()
            # Under the hood llama-cpp-python state extraction
            raw_state = self.llm.save_state() 
            metrics["extract_seconds"] = time.perf_counter() - t_start_extract
            
            # Wipe state to free VRAM before reload
            del self.llm
            import gc
            gc.collect()
            
        # 2. Measure Model Reload & Layout Re-allocation Time
        t_start_reload = time.perf_counter()
        target_layers = self._map_mode_to_layers(mode)
        self._load_model(target_layers) # Your internal llama_context builder
        metrics["reload_seconds"] = time.perf_counter() - t_start_reload
        
        # 3. Measure KV Injection Time into the new VRAM topology
        if 'raw_state' in locals():
            t_start_inject = time.perf_counter()
            self.llm.load_state(raw_state)
            metrics["inject_seconds"] = time.perf_counter() - t_start_inject

        self.current_strategy = mode
        return {"success": True, "metrics": metrics}