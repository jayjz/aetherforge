import os
import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.cache_manager import AetherCacheManager
from src.config import settings

try:
    from src.inference_engine import AetherEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

class EconomicGatekeeper:
    """
    Mathematically validates if a VRAM swap is profitable.
    Uses Exponential Moving Average (EMA) to learn actual hardware speeds in real-time.
    """
    def __init__(self):
        self.swap_penalty_seconds = settings.swap_penalty_seconds 
        self.alpha = 0.3  # EMA smoothing factor. 30% new measurement, 70% historical.
        
        # Initialize profiles with config defaults, but track live EMA separately
        self.profiles = {
            "high_fidelity": {"decode_tps": settings.tps_high_fidelity, "live_tps": settings.tps_high_fidelity},
            "balanced": {"decode_tps": settings.tps_balanced, "live_tps": settings.tps_balanced},
            "aggressive_quant": {"decode_tps": settings.tps_aggressive_quant, "live_tps": settings.tps_aggressive_quant}
        }

    def update_profile(self, mode: str, measured_tps: float):
        """Feeds live generation metrics back into the Gatekeeper's math with strict clamping."""
        if mode not in self.profiles or measured_tps <= 0:
            return
            
        current_ema = self.profiles[mode]["live_tps"]
        
        # Calculate raw EMA
        raw_ema = (measured_tps * settings.telemetry_alpha) + (current_ema * (1.0 - settings.telemetry_alpha))
        
        # Clamp the result to prevent transient anomalies from breaking the logic
        clamped_ema = max(settings.tps_min_clamp, min(settings.tps_max_clamp, raw_ema))
        
        self.profiles[mode]["live_tps"] = clamped_ema
        
        print(f"[Gatekeeper] Telemetry updated for '{mode}': {measured_tps:.2f} t/s -> EMA: {clamped_ema:.2f} t/s")

    def evaluate_swap(self, current_mode: str, target_mode: str, context_tokens: int, expected_output: int) -> bool:
        if current_mode == target_mode:
            return False 
            
        # ALWAYS use the live EMA for calculations
        current_tps = self.profiles.get(current_mode, self.profiles["balanced"])["live_tps"]
        target_tps = self.profiles.get(target_mode, self.profiles["balanced"])["live_tps"]
        
        time_to_stay = expected_output / current_tps
        
        state_io_penalty = settings.state_io_base_seconds + (context_tokens * settings.state_io_per_token_seconds)
        target_generation_time = expected_output / target_tps
        
        time_to_swap = self.swap_penalty_seconds + state_io_penalty + target_generation_time
        
        print(f"\n[Gatekeeper] Evaluation Request: {current_mode.upper()} -> {target_mode.upper()}")
        print(f" -> Active Context size: {context_tokens} tokens | Anticipated Output: {expected_output} tokens")
        print(f" -> Live TPS baselines - Current: {current_tps:.2f} t/s | Target: {target_tps:.2f} t/s")
        print(f" -> Estimated time to STAY: {time_to_stay:.2f}s")
        print(f" -> Estimated time to SWAP: {time_to_swap:.2f}s")
        
        if time_to_swap >= time_to_stay:
            print(" -> [REJECTED] Swap canceled. Latency deficit exceeds throughput dividend.")
            return False
            
        print(" -> [APPROVED] Hardware swap initiated.")
        return True
            
        current = self.profiles.get(current_mode, self.profiles["balanced"])
        target = self.profiles.get(target_mode, self.profiles["balanced"])
        
        time_to_stay = expected_output / current["decode_tps"]
        
        # IO penalty scaling pulled dynamically from settings
        state_io_penalty = settings.state_io_base_seconds + (context_tokens * settings.state_io_per_token_seconds)
        target_generation_time = expected_output / target["decode_tps"]
        
        time_to_swap = self.swap_penalty_seconds + state_io_penalty + target_generation_time
        
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
hypervisor_cache = AetherCacheManager(
    vram_budget_mb=settings.vram_budget_mb, 
    ram_budget_mb=settings.ram_budget_mb
)
hardware_engine = None
current_strategy = "balanced"
gatekeeper = EconomicGatekeeper()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global hardware_engine
    print("\n[API] Booting AetherForge Control Plane...")
    
    topo_path = "moe_topology.json"
    if os.path.exists(topo_path):
        hypervisor_cache.load_topology(topo_path)

    # STRICT STRANGER TEST VALIDATION: Fail early, fail visibly.
    if not os.path.exists(settings.model_path):
        critical_msg = (
            f"\n[CRITICAL ERROR] AetherForge cannot boot. Model file not found at: '{settings.model_path}'\n"
            f"Please verify your local path configuration or populate your .env file using .env.example."
        )
        print(critical_msg)
        raise FileNotFoundError(critical_msg)
    
    if HAS_ENGINE:
        try:
            print(f"[API] Hardware Engine active. Targeting model: {settings.model_path}")
            hardware_engine = AetherEngine(
                model_path=settings.model_path, 
                vram_budget_mb=settings.vram_budget_mb,
                n_ctx=settings.n_ctx
            )
        except Exception as e:
            print(f"[API] Fatal initialization crash on CUDA layer: {e}")
            raise RuntimeError(f"Hardware initialization failed: {e}")
    else:
        print("[API] Running in Brain-Only Simulation Mode. (Inference libraries uninstalled).")
        
    yield
    print("\n[API] Shutting down AetherForge Control Plane...")

# Version bumped to match Phase A deliverable state
app = FastAPI(title="AetherForge Hypervisor API", version="0.4.0", lifespan=lifespan)

# --- PYDANTIC SCHEMAS ---
class StrategyPayload(BaseModel):
    mode: str = Field(
        ..., 
        description="The VRAM strategy: 'high_fidelity' (max layers for coding/reasoning), 'balanced' (standard chat), or 'aggressive_quant' (summarization/routing)."
    )
    priority_layers: Optional[List[int]] = Field(
        default=[], 
        description="Specific layer indices to lock into VRAM. Leave empty unless specifically required."
    )
    estimated_context_tokens: Optional[int] = Field(
        500, 
        description="Estimated token size of the current context history."
    )
    expected_output_tokens: Optional[int] = Field(
        300, 
        description="Estimated number of tokens the upcoming generation will require."
    )
    context_text: Optional[str] = Field(
        None, 
        description="The exact raw prompt or conversation history. If provided, the Gatekeeper uses this for exact token counting."
    )

class GenerationPayload(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    strategy: Optional[StrategyPayload] = None
    simulate: bool = False

# --- ROUTES ---
@app.get("/system/cache")
async def get_cache_status():
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

@app.get("/system/tools")
async def get_tool_schema():
    """
    Exports AetherForge capabilities as an OpenAI-compatible function schema.
    Generated dynamically from the StrategyPayload Pydantic model to guarantee zero drift.
    """
    base_schema = StrategyPayload.model_json_schema()
    properties = base_schema.get("properties", {})
    for prop in properties.values():
        prop.pop("title", None)
        prop.pop("default", None)

    return {
        "type": "function",
        "function": {
            "name": "aetherforge_optimize_vram",
            "description": (
                "Hypervisor control: Dynamically allocates physical VRAM layers based on upcoming task complexity. "
                "Call this BEFORE executing heavy reasoning, coding, or summarization tasks to optimize tokens/sec. "
                "The hypervisor will mathematically validate the swap to prevent context latency penalties."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": base_schema.get("required", [])
            }
        }
    }

@app.post("/system/strategy")
async def update_strategy(payload: StrategyPayload):
    global current_strategy
    target_mode = payload.mode.lower()
    
    context_size = payload.estimated_context_tokens
    if payload.context_text and hardware_engine:
        context_size = hardware_engine.count_tokens(payload.context_text)
        print(f"[API] Tokenizer override: Agent text history dynamically counted at {context_size} tokens.")
    
    is_profitable = gatekeeper.evaluate_swap(
        current_mode=current_strategy,
        target_mode=target_mode,
        context_tokens=context_size,
        expected_output=payload.expected_output_tokens
    )
    
    if not is_profitable:
        return {
            "status": "rejected",
            "reason": "Swap latency overhead exceeds raw throughput gains.",
            "active_mode": current_strategy
        }
    
    print(f"\n[API] OVERRIDE: Gatekeeper approved strategy swap -> {target_mode.upper()}")
        
    current_strategy = target_mode
    
    if hardware_engine:
        success = hardware_engine.apply_strategy(current_strategy)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to apply hardware strategy.")
        return {"status": "hardware_strategy_applied", "active_mode": current_strategy}
    else:
        return {"status": "simulation_strategy_accepted", "active_mode": current_strategy}

@app.post("/generate")
async def generate_text(payload: GenerationPayload):
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
        if hardware_engine.current_strategy != active_mode:
            exact_prompt_tokens = hardware_engine.count_tokens(payload.prompt)
            print(f"[API] Engaging hardware layers under '{active_mode}' strategy constraints...")
        output = hardware_engine.generate(prompt=payload.prompt, max_tokens=payload.max_tokens)
        
        # --- CLOSE THE FEEDBACK LOOP ---
        measured_tps = output["metrics"].get("tokens_per_second", 0)
        if measured_tps > 0:
            gatekeeper.update_profile(active_mode, measured_tps)
        
        return {
            "text": output["text"],
            "metrics": output["metrics"],
            "hardware_engaged": True
        }
            
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