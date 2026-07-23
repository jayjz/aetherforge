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
from src.engines.base import BaseAetherEngine

class LlamaEngine(BaseAetherEngine):    
    def __init__(self, model_path: str, vram_budget_mb: float = 8000, n_ctx: int = 4096):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model_path = model_path
        self.vram_budget_mb = vram_budget_mb
        self.n_ctx = n_ctx
        self.current_strategy = "balanced"
        self.llm = None
        
        # Build memory profile dynamically from centralized config
        self.strategy_map = {
            "high_fidelity": settings.layers_high_fidelity,
            "balanced": settings.layers_balanced,
            "aggressive_quant": settings.layers_aggressive_quant
        }
        
        print(f"[Engine] Booting AetherForge Inference Core...")
        print(f"[Engine] Target Model: {os.path.basename(model_path)}")
        print(f"[Engine] Strategy Layer Scheme: {self.strategy_map}")
        
        # Initial boot
        self._load_model(self._map_mode_to_layers(self.current_strategy))
        print("[Engine] CUDA Engine Online.")

    def _map_mode_to_layers(self, mode: str) -> int:
        """Safely resolves the strategy to a specific layer count."""
        return self.strategy_map.get(mode, self.strategy_map["balanced"])

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

    def apply_strategy(self, mode: str) -> Dict[str, Any]:
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
            raw_state = self.llm.save_state() 
            metrics["extract_seconds"] = time.perf_counter() - t_start_extract
            
            # Wipe state to free VRAM before reload
            del self.llm
            gc.collect()
            
        # 2. Measure Model Reload & Layout Re-allocation Time
        t_start_reload = time.perf_counter()
        target_layers = self._map_mode_to_layers(mode)
        self._load_model(target_layers)
        metrics["reload_seconds"] = time.perf_counter() - t_start_reload
        
        # 3. Measure KV Injection Time into the new VRAM topology
        if 'raw_state' in locals():
            t_start_inject = time.perf_counter()
            self.llm.load_state(raw_state)
            metrics["inject_seconds"] = time.perf_counter() - t_start_inject

        self.current_strategy = mode
        return {"success": True, "metrics": metrics}

    def generate(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """
        Executes physical inference and calculates Gatekeeper telemetry.
        """
        if not self.llm:
            raise RuntimeError("Cannot generate: Hardware engine is offline.")

        t_start = time.perf_counter()
        
        # llama-cpp-python native inference call
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        t_end = time.perf_counter()
        elapsed_seconds = t_end - t_start
        
        text_result = output["choices"][0]["text"]
        
        # Safely extract token usage, fallback to tokenizer if missing
        generated_tokens = output.get("usage", {}).get("completion_tokens", 0)
        if generated_tokens == 0:
            generated_tokens = len(self.llm.tokenize(text_result.encode('utf-8')))
            
        tps = generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0

        return {
            "text": text_result,
            "metrics": {
                "tokens_generated": generated_tokens,
                "time_seconds": elapsed_seconds,
                "tokens_per_second": tps
            }
        }