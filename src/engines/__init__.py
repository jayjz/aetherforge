import os
from .base import BaseAetherEngine
from .mock_engine import MockAetherEngine
from .kt_engine import KTransformersEngine
from .llama_engine import LlamaEngine 

def create_engine(engine_type: str, model_path: str, vram_budget_mb: int, n_ctx: int) -> BaseAetherEngine:
    """
    Factory routes to the correct hardware backend.
    Includes defensive fallbacks to protect the control plane.
    """
    if engine_type == "mock":
        return MockAetherEngine(model_path, vram_budget_mb, n_ctx)
        
    elif engine_type == "ktransformers":
        if os.getenv("ENABLE_KTRANSFORMERS", "").lower() == "true":
            print("[Factory] ktransformers flag detected. Routing to KT Engine.")
            return KTransformersEngine(model_path, vram_budget_mb, n_ctx)
        else:
            print("[Factory] ENABLE_KTRANSFORMERS is not true. Falling back to LlamaEngine.")
            return LlamaEngine(model_path, vram_budget_mb, n_ctx)
            
    elif engine_type == "llama":
        return LlamaEngine(model_path, vram_budget_mb, n_ctx)
        
    else:
        raise ValueError(f"Unknown engine type requested: {engine_type}")