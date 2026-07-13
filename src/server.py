from fastapi import FastAPI, HTTPException
from .api_models import InferenceRequest, CacheStatusResponse, HypervisorStrategy, ExpertState
from .cache_manager import AetherCacheManager # Importing the brain we built in Route 1

app = FastAPI(title="AetherForge Hypervisor API", version="0.1.0")

# Initialize a dummy cache manager for the API to talk to (8GB VRAM, 32GB RAM)
manager = AetherCacheManager(vram_budget_mb=8000, ram_budget_mb=32000)

@app.get("/")
async def root():
    return {"message": "AetherForge Hypervisor is online."}

@app.get("/system/cache", response_model=CacheStatusResponse)
async def get_cache_status():
    """Allows an agent to check how choked the VRAM is before sending a massive task."""
    experts_in_vram = [
        ExpertState(expert_id=e.expert_id, location=e.location, precision=e.precision)
        for e in manager.experts.values() if e.location == "VRAM"
    ]
    
    return CacheStatusResponse(
        vram_budget_mb=manager.vram_budget_mb,
        current_vram_usage_mb=manager.current_vram_usage_mb,
        active_experts_in_vram=experts_in_vram,
        status="healthy" if manager.current_vram_usage_mb < manager.vram_budget_mb * 0.9 else "constrained"
    )

@app.post("/system/strategy")
async def set_global_strategy(strategy: HypervisorStrategy):
    """Allows the orchestrator to shift the hypervisor's behavior on the fly."""
    # In a real implementation, this would adjust the HOBBIT weights in the CacheManager
    print(f"[API] Agent requested strategy shift to: {strategy.mode}")
    return {"status": "success", "active_strategy": strategy.mode}

@app.post("/generate")
async def generate_text(request: InferenceRequest):
    """The main endpoint. Currently mocks generation while logging hypervisor intent."""
    
    if request.strategy:
        print(f"[API] Applying temporary strategy for this request: {request.strategy.mode}")
        
    # --- HERE IS WHERE ROUTE 1 MEETS ROUTE 3 ---
    # In the future, this endpoint will pass the prompt to llama-cpp-python.
    # As llama.cpp routes tokens, it will hit our CacheManager to prefetch experts.
    # For now, we mock a 2-second generation delay.
    
    import asyncio
    await asyncio.sleep(2) 
    
    return {
        "text": f"[Mock Generation] AetherForge received prompt: '{request.prompt[:20]}...'",
        "tokens_generated": 42,
        "vram_status_during_gen": "stable"
    }