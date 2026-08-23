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
        from .mock_engine import MockAetherEngine
        return MockAetherEngine(model_path, vram_budget_mb, n_ctx)
        
    elif engine_type == "llama":
        from .llama_engine import LlamaEngine
        return LlamaEngine(model_path, vram_budget_mb, n_ctx)
        
    elif engine_type == "ktransformers":
        # Stub kept for future MoE research
        from .kt_engine import KTransformersEngine
        return KTransformersEngine(model_path, vram_budget_mb, n_ctx)
        
    else:
        raise ValueError(f"Unknown AETHER_ENGINE type: {engine_type}")