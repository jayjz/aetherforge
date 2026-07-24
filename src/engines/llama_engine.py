"""
AetherForge Hardware Engine (The Muscle)
========================================
Manages physical execution of LLM generation via llama.cpp.
Implements hard VRAM barriers and C-level CUDA synchronization 
to prevent OOM races during Fast-Swaps.
"""

import os
import time
import gc
import ctypes
from typing import Dict, Any, Optional
from llama_cpp import Llama
from src.config import settings
from src.engines.base import BaseAetherEngine

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

class LlamaEngine(BaseAetherEngine):    
    def __init__(self, model_path: str, vram_budget_mb: float = 8000, n_ctx: int = 4096):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[Engine] FATAL: Model not found at {model_path}")
            
        self.model_path = model_path
        self.vram_budget_mb = vram_budget_mb
        self.n_ctx = n_ctx
        self.current_strategy = "balanced"
        self.llm: Optional[Llama] = None
        
        self.strategy_map = {
            "high_fidelity": settings.layers_high_fidelity,
            "balanced": settings.layers_balanced,
            "aggressive_quant": settings.layers_aggressive_quant
        }

        # Initialize NVML for pre-flight safety checks
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.nvml_handle = None
        else:
            self.nvml_handle = None
        
        print(f"[Engine] Booting AetherForge Inference Core...")
        if not self._load_model(self._map_mode_to_layers(self.current_strategy)):
            raise RuntimeError("[Engine] FATAL: Initial model load failed.")
        print("[Engine] CUDA Engine Online.")

    def _force_cuda_sync(self):
        """
        Hard VRAM barrier. 
        Forces the NVIDIA driver to synchronize the context and flush freed memory.
        """
        try:
            # Attempt to load the CUDA driver API library dynamically
            try:
                cuda = ctypes.CDLL("nvcuda.dll")  # Windows
            except OSError:
                cuda = ctypes.CDLL("libcuda.so")  # Linux
            
            # cuCtxSynchronize blocks until the device has completed all preceding requests
            cuda.cuCtxSynchronize()
        except Exception as e:
            print(f"[Engine] WARNING: C-level CUDA sync failed. Relying on Python GC. ({e})")

    def _check_vram_headroom(self) -> bool:
        """
        Gross baseline check. Rejects swaps if the GPU is fundamentally suffocating.
        Cannot perfectly predict llama.cpp footprint, but prevents obvious OOMs.
        """
        if not self.nvml_handle:
            return True # Fly blind if NVML is offline
            
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            free_mb = mem_info.free / (1024 ** 2)
            # Require at least 500MB of free VRAM overhead before attempting any allocations
            if free_mb < 500:
                print(f"[Engine] Pre-flight abort: Only {free_mb:.1f}MB VRAM free.")
                return False
            return True
        except Exception:
            return True

    def _map_mode_to_layers(self, mode: str) -> int:
        return self.strategy_map.get(mode, self.strategy_map["balanced"])

    def _load_model(self, n_gpu_layers: int) -> bool:
        """Safely attempts to allocate the model. Returns False on failure."""
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False
            )
            return True
        except Exception as e:
            print(f"[Engine] ALLOCATION FAILURE: {e}")
            self.llm = None
            return False
            
    def count_tokens(self, text: str) -> int:
        if not self.llm:
            return 0
        return len(self.llm.tokenize(text.encode('utf-8')))

    def apply_strategy(self, mode: str) -> Dict[str, Any]:
        metrics = {"extract_seconds": 0.0, "reload_seconds": 0.0, "inject_seconds": 0.0}
        
        if mode == self.current_strategy and self.llm is not None:
            return {"success": True, "metrics": metrics}

        if not self._check_vram_headroom():
            return {"success": False, "metrics": metrics}
            
        raw_state = None
        if self.llm is not None:
            t_start_extract = time.perf_counter()
            raw_state = self.llm.save_state() 
            metrics["extract_seconds"] = time.perf_counter() - t_start_extract
            
            # 1. Destroy Python object
            del self.llm
            self.llm = None
            # 2. Force Python GC to delete the underlying C++ pointer
            gc.collect()
            # 3. Force NVIDIA driver to physically flush the VRAM
            self._force_cuda_sync()
            
        t_start_reload = time.perf_counter()
        target_layers = self._map_mode_to_layers(mode)
        
        # ATTEMPT ALLOCATION
        if not self._load_model(target_layers):
            print("[Engine] Swap failed. Attempting emergency recovery to safe mode...")
            self._force_cuda_sync()
            # Fallback to the lowest memory footprint to keep the API alive
            safe_layers = self.strategy_map["aggressive_quant"]
            if not self._load_model(safe_layers):
                raise RuntimeError("CRITICAL: GPU OOM Recovery failed. Engine is brain-dead.")
            self.current_strategy = "aggressive_quant"
            return {"success": False, "metrics": metrics}

        metrics["reload_seconds"] = time.perf_counter() - t_start_reload
        
        # INJECT STATE
        if raw_state is not None:
            t_start_inject = time.perf_counter()
            try:
                self.llm.load_state(raw_state)
            except Exception as e:
                print(f"[Engine] WARNING: KV Cache state injection failed: {e}. Proceeding with blank state.")
                # We do not crash here; a blank state is better than a dead engine, 
                # but the agent pays a prefill penalty.
            metrics["inject_seconds"] = time.perf_counter() - t_start_inject

        self.current_strategy = mode
        return {"success": True, "metrics": metrics}

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        if not self.llm:
            raise RuntimeError("Cannot generate: Hardware engine is offline.")

        t_start = time.perf_counter()
        
        # We DO NOT catch exceptions here. If inference fails (e.g. CUDA error mid-generation),
        # it must bubble up so the API returns 500. Catching it and returning fake metrics 
        # poisons the Gatekeeper's math.
        output = self.llm(prompt, max_tokens=max_tokens, temperature=temperature)
            
        t_end = time.perf_counter()
        elapsed_seconds = t_end - t_start
        
        text_result = output["choices"][0]["text"]
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