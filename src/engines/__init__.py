"""
AetherForge Engine Factory
==========================
Dynamically loads the requested backend (Llama, KTransformers, or Mock).
"""

from typing import Optional
from src.engines.base import BaseAetherEngine
from src.engines.mock_engine import MockAetherEngine

def create_engine(engine_type: str, model_path: str, vram_budget_mb: int, n_ctx: int) -> BaseAetherEngine:
    """
    Factory pattern to initialize the correct hardware abstraction.
    """
    if engine_type == "llama":
        try:
            from src.engines.llama_engine import LlamaEngine
            print(f"[Factory] Initializing primary Llama.cpp backend...")
            return LlamaEngine(model_path=model_path, vram_budget_mb=vram_budget_mb, n_ctx=n_ctx)
        except ImportError as e:
            print(f"[Factory] Llama backend unavailable ({e}). Defaulting to Mock.")
            return MockAetherEngine(model_path=model_path, vram_budget_mb=vram_budget_mb, n_ctx=n_ctx)
            
    elif engine_type == "ktransformers":
        try:
            from src.engines.kt_engine import KTransformersEngine
            print(f"[Factory] Initializing KTransformers experimental backend...")
            return KTransformersEngine(model_path=model_path, vram_budget_mb=vram_budget_mb, n_ctx=n_ctx)
        except ImportError as e:
            print(f"[Factory] KTransformers backend unavailable ({e}). Defaulting to Mock.")
            return MockAetherEngine(model_path=model_path, vram_budget_mb=vram_budget_mb, n_ctx=n_ctx)
            
    elif engine_type == "mock":
        print(f"[Factory] Initializing headless Mock backend...")
        return MockAetherEngine(model_path=model_path, vram_budget_mb=vram_budget_mb, n_ctx=n_ctx)
        
    else:
        raise ValueError(f"Unknown engine_type requested: {engine_type}")