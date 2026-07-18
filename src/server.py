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

from src.mock_engine import MockAetherEngine

# --- 1. HARDENED STATE CONTAINER ---
class HypervisorState:
    """
    Thread-safe container isolating the hypervisor runtime context from global module scope.
    Encapsulates memory boundaries, active strategies, and hardware bridges.
    """
    def __init__(self):
        self.cache_manager = AetherCacheManager(
            vram_budget_mb=settings.vram_budget_mb, 
            ram_budget_mb=settings.ram_budget_mb
        )
        self.gatekeeper = EconomicGatekeeper()
        self.hardware_engine = None
        self.current_strategy = "balanced"
        self.lock = asyncio.Lock()  # Prevents overlapping hardware swaps during concurrent agent calls


class EconomicGatekeeper:
    """
    Mathematically validates if a VRAM swap is profitable.
    Uses Exponential Moving Average (EMA) to learn actual hardware speeds in real-time.
    """
    def __init__(self):
        self.swap_penalty_seconds = settings.swap_penalty_seconds 
        self.alpha = settings.telemetry_alpha

        self.profiles = {
            "high_fidelity": {"decode_tps": settings.tps_high_fidelity, "live_tps": settings.tps_high_fidelity},
            "balanced": {"decode_tps": settings.tps_balanced, "live_tps": settings.tps_balanced},
            "aggressive_quant": {"decode_tps": settings.tps_aggressive_quant, "live_tps": settings.tps_aggressive_quant}
        }

    def update_profile(self, mode: str, measured_tps: float):
        if mode not in self.profiles or measured_tps <= 0:
            return
        current_ema = self.profiles[mode]["live_tps"]
        raw_ema = (measured_tps * self.alpha) + (current_ema * (1.0 - self.alpha))
        clamped_ema = max(settings.tps_min_clamp, min(settings.tps_max_clamp, raw_ema))
        self.profiles[mode]["live_tps"] = clamped_ema
        print(f"[Gatekeeper] Telemetry updated for '{mode}': {measured_tps:.2f} t/s -> EMA: {clamped_ema:.2f} t/s")

    def evaluate_swap(self, current_mode: str, target_mode: str, context_tokens: int, expected_output: int) -> bool:
        if current_mode == target_mode:
            return False 
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


# --- 2. STATE-AWARE LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[API] Booting AetherForge Control Plane...")
    
    # Instantiate state container and bind to app lifecycle
    state = HypervisorState()
    app.state.hypervisor = state

    topo_path = "moe_topology.json"
    if os.path.exists(topo_path):
        state.cache_manager.load_topology(topo_path)

    if not os.path.exists(settings.model_path) and HAS_ENGINE:
        critical_msg = f"[CRITICAL ERROR] Model file not found at: '{settings.model_path}'"
        print(critical_msg)
        raise FileNotFoundError(critical_msg)
    
    if HAS_ENGINE and os.path.exists(settings.model_path):
        try:
            print(f"[API] Hardware Engine active. Targeting model: {settings.model_path}")
            state.hardware_engine = AetherEngine(
                model_path=settings.model_path, 
                vram_budget_mb=settings.vram_budget_mb,
                n_ctx=settings.n_ctx
            )
        except Exception as e:
            print(f"[API] Fatal initialization crash on CUDA layer: {e}")
            raise RuntimeError(f"Hardware initialization failed: {e}")
    else:
        print("[API] WARNING: CUDA engine missing or model not found.")
        print("[API] Falling back to MockAetherEngine. Operating in headless simulation mode.")
        state.hardware_engine = MockAetherEngine(
            model_path=settings.model_path,
            vram_budget_mb=settings.vram_budget_mb,
            n_ctx=settings.n_ctx
        )
        
    yield
    print("\n[API] Shutting down AetherForge Control Plane...")

app = FastAPI(title="AetherForge Hypervisor API", version="0.4.2", lifespan=lifespan)


# --- PYDANTIC SCHEMAS ---
class StrategyPayload(BaseModel):
    mode: str = Field(..., description="The VRAM strategy: 'high_fidelity', 'balanced', or 'aggressive_quant'.")
    priority_layers: Optional[List[int]] = Field(default=[])
    estimated_context_tokens: Optional[int] = Field(500)
    expected_output_tokens: Optional[int] = Field(300)
    context_text: Optional[str] = Field(None)

class GenerationPayload(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    strategy: Optional[StrategyPayload] = None
    simulate: bool = False


# --- 3. HARDENED ROUTES ---

@app.get("/system/cache")
async def get_cache_status(request: Request):
    state = request.app.state.hypervisor
    vram_experts = [e.expert_id for e in state.cache_manager.experts.values() if e.location == "VRAM"]
    
    return {
        "status": "online" if HAS_ENGINE else "simulation",
        "current_step": state.cache_manager.current_step,
        "vram_budget_mb": state.cache_manager.vram_budget_mb,
        "current_vram_usage_mb": state.cache_manager.current_vram_usage,
        "active_experts_in_vram": vram_experts,
        "active_strategy": state.hardware_engine.current_strategy,
        "engine_available": HAS_ENGINE
    }

@app.get("/system/metrics")
async def get_metrics(request: Request):
    state = request.app.state.hypervisor
    gatekeeper_telemetry = {
        mode: {
            "live_tps": profile["live_tps"],
            "baseline_tps": profile["decode_tps"]
        }
        for mode, profile in state.gatekeeper.profiles.items()
    }
    
    return {
        "timestamp": time.time(),
        "active_strategy": state.current_strategy,
        "vram_pressure": {
            "current_mb": state.cache_manager.current_vram_usage,
            "budget_mb": state.cache_manager.vram_budget_mb,
            "utilization_pct": (state.cache_manager.current_vram_usage / state.cache_manager.vram_budget_mb) * 100
        },
        "performance_baselines": gatekeeper_telemetry,
        "engine_state": "online" if HAS_ENGINE else "simulation"
    }

@app.get("/system/tools")
async def get_tool_schema():
    base_schema = StrategyPayload.model_json_schema()
    properties = base_schema.get("properties", {})
    for prop in properties.values():
        prop.pop("title", None)
        prop.pop("default", None)

    return {
        "type": "function",
        "function": {
            "name": "aetherforge_optimize_vram",
            "description": "Hypervisor control: Dynamically allocates physical VRAM layers based on task complexity.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": base_schema.get("required", [])
            }
        }
    }

@app.post("/system/strategy")
async def update_strategy(payload: StrategyPayload, request: Request):
    state = request.app.state.hypervisor
    target_mode = payload.mode.lower()
    
    # Enforce atomic swap isolation to defeat concurrent collision
    async with state.lock:
        context_size = payload.estimated_context_tokens
        if payload.context_text:
            context_size = state.hardware_engine.count_tokens(payload.context_text)
            print(f"[API] Tokenizer override: Context dynamically counted at {context_size} tokens.")
        
        is_profitable = state.gatekeeper.evaluate_swap(
            current_mode=state.current_strategy,
            target_mode=target_mode,
            context_tokens=context_size,
            expected_output=payload.expected_output_tokens
        )
        
        if not is_profitable:
            return {
                "status": "rejected",
                "reason": "Swap latency overhead exceeds raw throughput gains.",
                "active_mode": state.current_strategy
            }
        
        print(f"\n[API] OVERRIDE: Gatekeeper approved strategy swap -> {target_mode.upper()}")
        state.current_strategy = target_mode
        success = state.hardware_engine.apply_strategy(state.current_strategy)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to apply hardware strategy.")
            
        return {"status": "strategy_applied", "active_mode": state.current_strategy}

@app.post("/generate")
async def generate_text(payload: GenerationPayload, request: Request):
    state = request.app.state.hypervisor
    active_mode = payload.strategy.mode if payload.strategy else state.current_strategy

    async with state.lock:
        if state.hardware_engine.current_strategy != active_mode:
            exact_prompt_tokens = state.hardware_engine.count_tokens(payload.prompt)
            print(f"[API] JIT Tokenizer verification: Generation prompt measured at {exact_prompt_tokens} tokens.")
            
            is_profitable = state.gatekeeper.evaluate_swap(
                current_mode=state.current_strategy,
                target_mode=active_mode,
                context_tokens=exact_prompt_tokens,
                expected_output=payload.max_tokens
            )
            
            if is_profitable:
                state.hardware_engine.apply_strategy(active_mode)
                state.current_strategy = active_mode
            else:
                print(f"[API] JIT Swap mathematically rejected. Enforcing active state execution [{state.current_strategy.upper()}].")
                active_mode = state.current_strategy

        print(f"[API] Engaging hardware layers under '{active_mode}' strategy constraints...")
        output = state.hardware_engine.generate(prompt=payload.prompt, max_tokens=payload.max_tokens)

    # Telemetry updates slip outside the atomic hardware execution lock to protect concurrency throughput
    measured_tps = output["metrics"].get("tokens_per_second", 0)
    if measured_tps > 0:
        state.gatekeeper.update_profile(active_mode, measured_tps)

    return {
        "text": output["text"],
        "metrics": output["metrics"],
        "hardware_engaged": HAS_ENGINE
    }