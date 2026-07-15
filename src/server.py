import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.cache_manager import AetherCacheManager

# Graceful import to prevent Linux environments from crashing if llama-cpp-python isn't optimized
try:
    from src.inference_engine import AetherEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

# --- GLOBAL STATE ---
# Initialize the Brain
hypervisor_cache = AetherCacheManager(vram_budget_mb=8000, ram_budget_mb=32000)
hardware_engine = None
current_strategy = "balanced"

# --- MODERN LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global hardware_engine
    print("\n[API] Booting AetherForge Control Plane...")
    
    # 1. Load Topology into the Brain
    topo_path = "moe_topology.json"
    if os.path.exists(topo_path):
        hypervisor_cache.load_topology(topo_path)
    else:
        print("[!] Warning: moe_topology.json not found. Run model_analyzer.py first.")

    # 2. Attempt to initialize the Hardware Engine (The Muscle)
    model_path = r"models\DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
    
    if HAS_ENGINE and os.path.exists(model_path):
        try:
            print("[API] Hardware Engine detected. Preparing physical tensor bridge...")
            hardware_engine = AetherEngine(model_path=model_path, vram_budget_mb=8000)
            print("[API] RTX 4060 Locked and Loaded. Waiting for Agent Commands.")
        except Exception as e:
            print(f"[API] Hardware boot failed. Error: {e}")
            hardware_engine = None
    else:
        print("[API] Running in Brain-Only Simulation Mode. Hardware Engine bypassed.")
        
    yield  # Application runs while yielding here
    
    # 3. Clean Shutdown logic would go here
    print("\n[API] Shutting down AetherForge Control Plane...")


app = FastAPI(title="AetherForge Hypervisor API", version="0.3.0", lifespan=lifespan)

# --- PYDANTIC SCHEMAS ---
class StrategyPayload(BaseModel):
    mode: str = Field(..., description="The requested VRAM allocation mode.")
    priority_layers: Optional[List[int]] = []

class GenerationPayload(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    strategy: Optional[StrategyPayload] = None
    simulate: bool = False  # Set to False so Windows naturally engages the GPU

# --- ROUTES ---
@app.get("/system/cache")
async def get_cache_status():
    """Returns the live state of the predictive hypervisor."""
    vram_experts = [e.expert_id for e in hypervisor_cache.experts.values() if e.location == "VRAM"]
    
    return {
        "status": "online" if hardware_engine else "simulation",
        "current_step": hypervisor_cache.current_step,
        "vram_budget_mb": hypervisor_cache.vram_budget_mb,
        "current_vram_usage_mb": hypervisor_cache.current_vram_usage,
        "active_experts_in_vram": vram_experts,
        "active_strategy": hardware_engine.current_strategy if hardware_engine else current_strategy,
        "engine_available": hardware_engine is not None
    }

@app.post("/system/strategy")
async def update_strategy(payload: StrategyPayload):
    """The Agent-to-Engine Bridge: Injects new memory management rules mid-task."""
    global current_strategy
    current_strategy = payload.mode
    
    print(f"\n[API] OVERRIDE: Agent injected new VRAM strategy -> {current_strategy.upper()}")
    
    if payload.priority_layers:
        print(f"[API] Locking layers into priority retention: {payload.priority_layers}")
        
    # Physically execute the VRAM fast-swap if hardware is available
    if hardware_engine:
        success = hardware_engine.apply_strategy(current_strategy)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to apply hardware strategy.")
        return {"status": "hardware_strategy_applied", "active_mode": current_strategy}
    else:
        return {"status": "simulation_strategy_accepted", "active_mode": current_strategy}

@app.post("/generate")
async def generate_text(payload: GenerationPayload):
    """Executes inference while actively applying the injected strategy."""
    global current_strategy # <---FIX SCOPING ISSUE: Ensure the global variable is referenced correctly

    # Just-in-time strategy check if the agent passed it in the generation payload
    active_mode = payload.strategy.mode if payload.strategy else current_strategy
    
    if payload.simulate or not hardware_engine:
        print(f"[API] Simulating token routing under '{active_mode}' strategy constraints...")
        
        # Simulate the cache juggling a few tokens to test the Brain's logic
        import random
        for _ in range(5):
            active = [(random.randint(0, 63), 0.8), (random.randint(0, 63), 0.2)]
            hypervisor_cache.route_token(layer_index=0, active_experts=active)
            
        await asyncio.sleep(1) # Fake inference delay
        return {
            "text": "[SIMULATED RESPONSE] AetherForge logic successfully routed.",
            "metrics": {
                "tokens_generated": payload.max_tokens,
                "active_strategy": active_mode
            },
            "hardware_engaged": False
        }
        
    else:
        # THE REAL HARDWARE EXECUTION
        print(f"[API] Engaging RTX 4060 under '{active_mode}' strategy constraints...")
        
        # Failsafe: Ensure the engine is actually in the requested mode before generating
        if hardware_engine.current_strategy != active_mode:
            hardware_engine.apply_strategy(active_mode)
            current_strategy = active_mode
        
        # Capture the actual generated text and metrics from your custom engine
        output = hardware_engine.generate(prompt=payload.prompt, max_tokens=payload.max_tokens)
        
        return {
            "text": output["text"],
            "metrics": output["metrics"],
            "hardware_engaged": True
        }