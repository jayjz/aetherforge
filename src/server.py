import os
import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.cache_manager import AetherCacheManager
from src.config import settings
from src.engines import create_engine

# --- OPTIONAL DEFENSIBLY LOADED NVML TOOLING ---
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class HardwareMonitor:
    """
    Reads physical hardware vitals using NVIDIA Management Library.
    Degrades gracefully to safe simulated modes if missing or on headless targets.
    """
    def __init__(self):
        self.active = False
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                # Targets primary consumer GPU instance (RTX 4060)
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.active = True
                print("[HardwareMonitor] NVML hardware bindings successfully mapped. Silicon telemetry active.")
            except Exception as e:
                print(f"[HardwareMonitor] NVML initialization bypassed: {e}. Running simulation fallback.")
        else:
            print("[HardwareMonitor] nvidia-ml-py library absent. Running simulation fallback.")

    def get_vitals(self) -> Dict[str, Any]:
        if not self.active:
            return {"temp_c": 0, "vram_pct": 0.0, "status": "simulated"}
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            vram_pct = (mem_info.used / mem_info.total) * 100.0
            return {"temp_c": temp, "vram_pct": vram_pct, "status": "online"}
        except Exception as e:
            print(f"[HardwareMonitor] Silicon polling telemetry interrupt: {e}")
            return {"temp_c": 0, "vram_pct": 0.0, "status": "error"}

    def shutdown(self):
        if self.active:
            try:
                pynvml.nvmlShutdown()
                print("[HardwareMonitor] NVML resource contexts cleanly unmapped.")
            except:
                pass


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
        self.hardware_monitor = HardwareMonitor()
        self.hardware_engine = None
        self.current_strategy = "balanced"
        self.is_simulated = True  # Defaults to true, overridden by lifespan factory
        self.lock = asyncio.Lock()  # Prevents overlapping hardware swaps during concurrent agent calls


class EconomicGatekeeper:
    """
    Mathematically validates if a VRAM swap is profitable.
    Learns throughput (TPS) and physical state serialization costs via EMA.
    """
    def __init__(self):
        # We start with the calibrated defaults from settings, but learn live values
        self.live_swap_penalty = settings.swap_penalty_seconds 
        self.live_io_penalty = settings.state_io_base_seconds
        self.alpha = settings.telemetry_alpha

        self.profiles = {
            "high_fidelity": {"decode_tps": settings.tps_high_fidelity, "live_tps": settings.tps_high_fidelity},
            "balanced": {"decode_tps": settings.tps_balanced, "live_tps": settings.tps_balanced},
            "aggressive_quant": {"decode_tps": settings.tps_aggressive_quant, "live_tps": settings.tps_aggressive_quant}
        }

    def update_hardware_latencies(self, physical_swap_seconds: float, io_seconds: float):
        """Updates the learned base cost of moving data across buses."""
        if physical_swap_seconds > 0:
            self.live_swap_penalty = (physical_swap_seconds * self.alpha) + (self.live_swap_penalty * (1.0 - self.alpha))
        if io_seconds > 0:
            self.live_io_penalty = (io_seconds * self.alpha) + (self.live_io_penalty * (1.0 - self.alpha))
            
        print(f"[Gatekeeper] Telemetry updated -> Learned Swap: {self.live_swap_penalty:.2f}s | Learned IO: {self.live_io_penalty:.2f}s")

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

        # --- HARDWARE SAFETY GUARD ---
        if context_tokens > settings.max_safe_context_tokens:
            print(f" -> [REJECTED - SAFETY GUARD] Context size ({context_tokens}) exceeds max safe swap limit ({settings.max_safe_context_tokens}).")
            print(" -> Preventing catastrophic VRAM thrash/OOM. Forcing active state execution.")
            return False

        current_tps = self.profiles.get(current_mode, self.profiles["balanced"])["live_tps"]
        target_tps = self.profiles.get(target_mode, self.profiles["balanced"])["live_tps"]
        
        time_to_stay = expected_output / current_tps
        
        # Uses the dynamically learned base IO penalty + scaling token multiplier
        state_io_overhead = self.live_io_penalty + (context_tokens * settings.state_io_per_token_seconds)
        target_generation_time = expected_output / target_tps
        
        time_to_swap = self.live_swap_penalty + state_io_overhead + target_generation_time
        
        print(f"\n[Gatekeeper] Evaluation Request: {current_mode.upper()} -> {target_mode.upper()}")
        print(f" -> Active Context size: {context_tokens} tokens | Anticipated Output: {expected_output} tokens")
        print(f" -> Live Cost Baselines - Swap Penalty: {self.live_swap_penalty:.2f}s | Base IO: {self.live_io_penalty:.2f}s")
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

    # Determine target engine (Llama physical vs Headless Mock)
    target_engine = "llama" if os.path.exists(settings.model_path) else "mock"
    
    try:
        state.hardware_engine = create_engine(
            engine_type=target_engine,
            model_path=settings.model_path,
            vram_budget_mb=settings.vram_budget_mb,
            n_ctx=settings.n_ctx
        )
        state.is_simulated = state.hardware_engine.__class__.__name__ == "MockAetherEngine"
    except Exception as e:
        print(f"[API] Fatal Engine Factory crash: {e}")
        raise RuntimeError(f"Hypervisor engine initialization failed: {e}")
        
    yield
    print("\n[API] Shutting down AetherForge Control Plane...")
    state.hardware_monitor.shutdown()

app = FastAPI(title="AetherForge Hypervisor API", version="0.5.0", lifespan=lifespan)


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
        "status": "simulation" if state.is_simulated else "online",
        "current_step": state.cache_manager.current_step,
        "vram_budget_mb": state.cache_manager.vram_budget_mb,
        "current_vram_usage_mb": state.cache_manager.current_vram_usage,
        "active_experts_in_vram": vram_experts,
        "active_strategy": state.hardware_engine.current_strategy,
        "engine_available": not state.is_simulated
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
        "silicon_vitals": state.hardware_monitor.get_vitals(),
        "performance_baselines": gatekeeper_telemetry,
        "engine_state": "simulation" if state.is_simulated else "online"
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
    
    async with state.lock:
        # Prevent hardware strategy execution if the machine is running too hot
        vitals = state.hardware_monitor.get_vitals()
        if vitals["status"] == "online":
            if vitals["temp_c"] >= settings.max_gpu_temp_c:
                raise HTTPException(status_code=503, detail="Strategy modification aborted: Hardware thermal ceiling exceeded.")
            if vitals["vram_pct"] >= settings.max_vram_allocation_pct:
                raise HTTPException(status_code=503, detail="Strategy modification aborted: Critical hardware VRAM pressure limit reached.")

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
        
        # Execute swap and capture actual wall-clock metrics
        result = state.hardware_engine.apply_strategy(state.current_strategy)
        
        # Support handling dictionary returns from real engine or old booleans from stub configurations
        if isinstance(result, dict):
            success = result.get("success", False)
            swap_metrics = result.get("metrics", {})
        else:
            success = result
            swap_metrics = {}
            
        if not success:
            raise HTTPException(status_code=500, detail="Failed to apply hardware strategy.")
            
        # Extract empirical serialization overheads
        io_total = swap_metrics.get("extract_seconds", 0.0) + swap_metrics.get("inject_seconds", 0.0)
        reload_total = swap_metrics.get("reload_seconds", 0.0)
        
        if io_total > 0 or reload_total > 0:
            state.gatekeeper.update_hardware_latencies(
                physical_swap_seconds=reload_total, 
                io_seconds=io_total
            )
            
        return {"status": "strategy_applied", "active_mode": state.current_strategy}

@app.post("/generate")
async def generate_text(payload: GenerationPayload, request: Request):
    state = request.app.state.hypervisor
    
    # --- 1. PROACTIVE PREFLIGHT: Guard the engine context from heavy ingest loads ---
    exact_prompt_tokens = state.hardware_engine.count_tokens(payload.prompt)
    if exact_prompt_tokens > settings.max_safe_context_tokens:
        print(f"[PREFLIGHT FAILED] Agent execution rejected. {exact_prompt_tokens} tokens exceeds absolute ceiling of {settings.max_safe_context_tokens}.")
        raise HTTPException(
            status_code=413, 
            detail=f"Context payload size ({exact_prompt_tokens} tokens) violates the active hardware safety ceiling ({settings.max_safe_context_tokens} tokens)."
        )

    # --- 2. PROACTIVE PREFLIGHT: Assert silicon safety profiles before computation ---
    vitals = state.hardware_monitor.get_vitals()
    if vitals["status"] == "online":
        if vitals["temp_c"] >= settings.max_gpu_temp_c:
            print(f"[THERMAL OVERFLOW] Execution aborted. GPU Temp at {vitals['temp_c']}°C exceeds boundary ({settings.max_gpu_temp_c}°C).")
            raise HTTPException(status_code=503, detail="Generation rejected: Hardware thermal threshold exceeded. Backing off.")
        if vitals["vram_pct"] >= settings.max_vram_allocation_pct:
            print(f"[VRAM OVERFLOW] Execution aborted. Active footprint at {vitals['vram_pct']:.2f}% exceeds boundary ({settings.max_vram_allocation_pct}%).")
            raise HTTPException(status_code=503, detail="Generation rejected: Insufficient VRAM structural headroom to proceed safely.")

    active_mode = payload.strategy.mode if payload.strategy else state.current_strategy

    async with state.lock:
        if state.hardware_engine.current_strategy != active_mode:
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
        "hardware_engaged": not state.is_simulated
    }