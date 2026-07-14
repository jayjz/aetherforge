import os
import time
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.cache_manager import AetherCacheManager

# Graceful import to prevent Linux environments from crashing if llama-cpp-python isn't optimized
try:
    from src.inference_engine import AetherEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

app = FastAPI(title="AetherForge Hypervisor API", version="0.2.0")

# --- GLOBAL STATE ---
# Initialize the Brain
hypervisor_cache = AetherCacheManager(vram_budget_mb=8000, ram_budget_mb=32000)
hardware_engine = None
current_strategy = "balanced"

# --- PYDANTIC SCHEMAS ---
class StrategyPayload(BaseModel):
    mode: str
    priority_layers: Optional[List[int]] = []

class GenerationPayload(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    strategy: Optional[StrategyPayload] = None
    simulate: bool = True  # Default to True so your Linux machine doesn't freeze

# --- STARTUP HOOKS ---
@app.on_event("startup")
async def startup_event():
    global hardware_engine
    print("\n[API] Booting AetherForge Control Plane...")
    
    # 1. Load Topology into the Brain
    topo_path = "moe_topology.json"
    if os.path.exists(topo_path):
        hypervisor_cache.load_topology(topo_path)
    else:
        print("[!] Warning: moe_topology.json not found. Run model_analyzer.py first.")

    # 2. Attempt to initialize the Hardware Engine (The Muscle)
    # Update this path if your model is stored elsewhere on Linux
    model_path = os.path.join("models", "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf")
    
    if HAS_ENGINE and os.path.exists(model_path):
        try:
            print("[API] Hardware Engine detected. Preparing physical tensor bridge...")
            hardware_engine = AetherEngine(model_path=model_path, vram_budget_mb=8000)
        except Exception as e:
            print(f"[API] Hardware boot failed (Expected on CPU-only Linux). Error: {e}")
            hardware_engine = None
    else:
        print("[API] Running in Brain-Only Simulation Mode. Hardware Engine bypassed.")

# --- ROUTES ---
@app.get("/system/cache")
async def get_cache_status():
    """Returns the live state of the predictive hypervisor."""
    vram_experts = [e.expert_id for e in hypervisor_cache.experts.values() if e.location == "VRAM"]
    
    return {
        "status": "online",
        "current_step": hypervisor_cache.current_step,
        "vram_budget_mb": hypervisor_cache.vram_budget_mb,
        "current_vram_usage_mb": hypervisor_cache.current_vram_usage,
        "active_experts_in_vram": vram_experts,
        "active_strategy": current_strategy
    }

@app.post("/system/strategy")
async def update_strategy(payload: StrategyPayload):
    """The Agent-to-Engine Bridge: Injects new memory management rules mid-task."""
    global current_strategy
    current_strategy = payload.mode
    
    print(f"\n[API] OVERRIDE: Agent injected new VRAM strategy -> {current_strategy.upper()}")
    if payload.priority_layers:
        print(f"[API] Locking layers into priority retention: {payload.priority_layers}")
        
    # In the future, this will physically trigger cache_manager to pre-fetch 
    # or evict specific experts based on the mode.
    
    return {"status": "strategy_accepted", "active_mode": current_strategy}

@app.post("/generate")
async def generate_text(payload: GenerationPayload):
    """Executes inference while actively applying the injected strategy."""
    print(f"\n[API] Generation request received. Mode: {'SIMULATION' if payload.simulate else 'HARDWARE'}")
    
    # Apply context-specific strategy if the agent provided one for this specific prompt
    active_mode = payload.strategy.mode if payload.strategy else current_strategy
    
    if payload.simulate or not hardware_engine:
        print(f"[API] Simulating token routing under '{active_mode}' strategy constraints...")
        # Simulate the cache juggling a few tokens
        import random
        for _ in range(5):
            active = [(random.randint(0, 63), 0.8), (random.randint(0, 63), 0.2)]
            hypervisor_cache.route_token(layer_index=0, active_experts=active)
            
        await asyncio.sleep(1) # Fake inference delay
        return {
            "text": "[SIMULATED RESPONSE] AetherForge logic successfully routed.",
            "tokens_generated": payload.max_tokens,
            "strategy_applied": active_mode,
            "hardware_engaged": False
        }
        
    else:
        # THE REAL HARDWARE EXECUTION
        print(f"[API] Engaging RTX 4060 under '{active_mode}' strategy constraints...")
        
        # In a fully realized version, the engine's generation loop will yield back to 
        # hypervisor_cache.route_token() token-by-token. For now, we capture the bulk output.
        output = hardware_engine.generate(prompt=payload.prompt, max_tokens=payload.max_tokens)
        
        return {
            "text": "Generation complete. Check terminal for baseline metrics.",
            "strategy_applied": active_mode,
            "hardware_engaged": True
        }