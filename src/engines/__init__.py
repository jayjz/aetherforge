"""
AetherForge Engine Factory
==========================
Lazy-loads hardware engines to ensure the Mock control plane 
can boot on systems lacking CUDA or C++ bindings.
"""

def create_engine(engine_type: str, model_path: str, vram_budget_mb: float, n_ctx: int):
    engine_type = engine_type.lower()
    
    if engine_type == "mock":
        from .mock_engine import MockAetherEngine
        return MockAetherEngine(model_path, vram_budget_mb, n_ctx)
        
    elif engine_type == "llama":
        from .llama_engine import LlamaEngine
        return LlamaEngine(model_path, vram_budget_mb, n_ctx)
        
    elif engine_type == "ktransformers":
        # Stub kept for future MoE research
        from .ktransformers_engine import KTransformersEngine
        return KTransformersEngine(model_path, vram_budget_mb, n_ctx)
        
    else:
        raise ValueError(f"Unknown AETHER_ENGINE type: {engine_type}")