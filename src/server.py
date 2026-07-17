import os
import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.cache_manager import AetherCacheManager

# Graceful import to prevent Linux environments from crashing if llama-cpp-python isn't optimized
try:
    from src.inference_engine import AetherEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

# --- DETEKTED ECONOMIC GATEKEEPER ---
class EconomicGatekeeper:
    """
    Mathematically validates if a VRAM swap is profitable.
    Prevents high-overhead context vaporization (KV-Cache destruction)
    unless the token throughput dividend covers the cost.
    """
    def __init__(self):
        # Physical baseline time to tear down and rebuild a llama.cpp instance
        self.swap_penalty_seconds = 5.8 
        
        # Hardware profiling boundaries verified on local consumer hardware
        self.profiles = {
            "high_fidelity": {"decode_tps": 23.71, "prefill_tps": 1800},
            "balanced": {"decode_tps": 11.10, "prefill_tps": 400},
            "aggressive_quant": {"decode_tps": 12.10, "prefill_tps": 600}
        }

    def evaluate_swap(self, current_mode: str, target_mode: str, context_tokens: int, expected_output: int) -> bool:
        if current_mode == target_mode:
            return False  # Already in the optimal layout
            
        current = self.profiles.get(current_mode, self.profiles["balanced"])
        target = self.profiles.get(target_mode, self.profiles["balanced"])
        
        # Scenario A: Remain in the current sub-optimal strategy
        # The KV-Cache is already warm, so prefill penalty is negligible.
        time_to_stay = expected_output / current["decode_tps"]
        
        # Scenario B: Fire a VRAM Fast-Swap
        # We pay the physical hardware swap overhead + cold KV-Cache prefill re-ingestion
        target_prefill_time = context_tokens / target["prefill_tps"]
        target_generation_time = expected_output / target["decode_tps"]
        time_to_swap = self.swap_penalty_seconds + target_prefill_time + target_generation_time
        
        print(f"\n[Gatekeeper] Evaluation Request: {current_mode.upper()} -> {target_mode.upper()}")
        print(f" -> Active Context size: {context_tokens} tokens | Anticipated Output: {expected_output} tokens")
        print(f" -> Estimated time to STAY: {time_to_stay:.2f}s")
        print(f" -> Estimated time to SWAP: {time_to_swap:.2f}s")
        
        if time_to_swap >= time_to_stay:
            print(" -> [REJECTED] Swap canceled. Latency deficit exceeds throughput dividend.")
            return False
            
        print(" -> [APPROVED] Hardware swap initiated.")
        return True

# --- GLOBAL STATE ---
hypervisor_cache = AetherCacheManager(vram_budget_mb=8000, ram_budget_mb=32000)
hardware_engine = None
current_strategy = "balanced"
gatekeeper = EconomicGatekeeper()

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
            print("[API] Consumer Hardware Acceleration Engine Active. Awaiting commands.")
        except Exception as e:
            print(f"[API] Hardware boot failed. Error: {e}")
            hardware_engine = None
    else:
        print("[API] Running in Brain-Only Simulation Mode. Hardware Engine bypassed.")
        
    yield
    print("\n[API] Shutting down AetherForge Control Plane...")

app = FastAPI(title="AetherForge Hypervisor API", version="0.3.0", lifespan=lifespan)

# --- PYDANTIC SCHEMAS ---
class StrategyPayload(BaseModel):
    mode: str = Field(..., description="The requested VRAM allocation mode.")
    priority_layers: Optional[List[int]] = []
    estimated_context_tokens: Optional[int] = Field(500, description="Size of current history context to re-evaluate.")
    expected_output_tokens: Optional[int] = Field(300, description="Agent's projected token generation size.")
    context_text: Optional[str] = Field(None, description="The actual text history. If provided, overrides estimates with exact math.")

class GenerationPayload(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    strategy: Optional[StrategyPayload] = None
    simulate: bool = False

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
    target_mode = payload.mode.lower()
    
    # Intercept and override agent estimates if exact text history is provided
    context_size = payload.estimated_context_tokens
    if payload.context_text and hardware_engine:
        context_size = hardware_engine.count_tokens(payload.context_text)
        print(f"[API] Tokenizer override: Agent text history dynamically counted at {context_size} tokens.")
    
    # Evaluate if switching strategies introduces an aggregate latency penalty
    is_profitable = gatekeeper.evaluate_swap(
        current_mode=current_strategy,
        target_mode=target_mode,
        context_tokens=context_size,
        expected_output=payload.expected_output_tokens
    )
    
    if not is_profitable:
        return {
            "status": "rejected",
            "reason": "Swap latency overhead and KV-Cache rebuild cost exceed raw throughput gains.",
            "active_mode": current_strategy
        }
    
    print(f"\n[API] OVERRIDE: Gatekeeper approved strategy swap -> {target_mode.upper()}")
    if payload.priority_layers:
        print(f"[API] Locking layers into priority retention: {payload.priority_layers}")
        
    current_strategy = target_mode
    
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
    global current_strategy

    active_mode = payload.strategy.mode if payload.strategy else current_strategy
    
    if payload.simulate or not hardware_engine:
        print(f"[API] Simulating token routing under '{active_mode}' strategy constraints...")
        import random
        for _ in range(5):
            active = [(random.randint(0, 63), 0.8), (random.randint(0, 63), 0.2)]
            hypervisor_cache.route_token(layer_index=0, active_experts=active)
            
        await asyncio.sleep(1)
        return {
            "text": "[SIMULATED RESPONSE] AetherForge logic successfully routed.",
            "metrics": {
                "tokens_generated": payload.max_tokens,
                "active_strategy": active_mode
            },
            "hardware_engaged": False
        }
        
    else:
        # Just-in-Time (JIT) Gatekeeper Interception
        if hardware_engine.current_strategy != active_mode:
            exact_prompt_tokens = hardware_engine.count_tokens(payload.prompt)
            print(f"[API] JIT Tokenizer verification: Generation prompt measured at {exact_prompt_tokens} tokens.")
            
            is_profitable = gatekeeper.evaluate_swap(
                current_mode=current_strategy,
                target_mode=active_mode,
                context_tokens=exact_prompt_tokens,
                expected_output=payload.max_tokens
            )
            
            if is_profitable:
                hardware_engine.apply_strategy(active_mode)
                current_strategy = active_mode
            else:
                print(f"[API] JIT Swap mathematically rejected. Enforcing active state execution [{current_strategy.upper()}].")
                active_mode = current_strategy
        
        print(f"[API] Engaging hardware layers under '{active_mode}' strategy constraints...")
        output = hardware_engine.generate(prompt=payload.prompt, max_tokens=payload.max_tokens)
        
        return {
            "text": output["text"],
            "metrics": output["metrics"],
            "hardware_engaged": True
        }