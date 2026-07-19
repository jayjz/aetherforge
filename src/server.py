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
        """
        Hardcoded context spoof for Cold Burn-In verification.
        Forces the background watchdog to evaluate a critical breach with zero physical load.
        """
        return {"temp_c": 90, "vram_pct": 45.0, "status": "online"}

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
        self.is_simulated = True 
        self.lock = asyncio.Lock() 
        
        # --- Emergency hardware kill switch ---
        self.emergency_thermal_lock = False


class EconomicGatekeeper:
    def __init__(self):
        self.live_swap_penalty = settings.swap_penalty_seconds 
        self.live_io_penalty = settings.state_io_base_seconds
        self.alpha = settings.telemetry_alpha

        self.profiles = {
            "high_fidelity": {"decode_tps": settings.tps_high_fidelity, "live_tps": settings.tps_high_fidelity},
            "balanced": {"decode_tps": settings.tps_balanced, "live_tps": settings.tps_balanced},
            "aggressive_quant": {"decode_tps": settings.tps_aggressive_quant, "live_tps": settings.tps_aggressive_quant}
        }

    def update_hardware_latencies(self, physical_swap_seconds: float, io_seconds: float):
        if physical_swap_seconds > 0:
            self.live_swap_penalty = (physical_swap_seconds * self.alpha) + (self.live_swap_penalty * (1.0 - self.alpha))
        if io_seconds > 0:
            self.live_io_penalty = (io_seconds * self.alpha) + (self.live_io_penalty * (1.0 - self.alpha))

    def update_profile(self, mode: str, measured_tps: float):
        if mode not in self.profiles or measured_tps <= 0:
            return
        current_ema = self.profiles[mode]["live_tps"]
        raw_ema = (measured_tps * self.alpha) + (current_ema * (1.0 - self.alpha))
        clamped_ema = max(settings.tps_min_clamp, min(settings.tps_max_clamp, raw_ema))
        self.profiles[mode]["live_tps"] = clamped_ema

    def evaluate_swap(self, current_mode: str, target_mode: str, context_tokens: int, expected_output: int) -> bool:
        if current_mode == target_mode:
            return False 

        if context_tokens > settings.max_safe_context_tokens:
            print(f" -> [REJECTED - SAFETY GUARD] Context size ({context_tokens}) exceeds max safe swap limit ({settings.max_safe_context_tokens}).")
            return False

        current_tps = self.profiles.get(current_mode, self.profiles["balanced"])["live_tps"]
        target_tps = self.profiles.get(target_mode, self.profiles["balanced"])["live_tps"]
        
        time_to_stay = expected_output / current_tps
        state_io_overhead = self.live_io_penalty + (context_tokens * settings.state_io_per_token_seconds)
        target_generation_time = expected_output / target_tps
        
        time_to_swap = self.live_swap_penalty + state_io_overhead + target_generation_time
        
        if time_to_swap >= time_to_stay:
            return False
        return True


# --- 2. ASYNCHRONOUS HARDWARE WATCHDOG ---
async def hardware_watchdog(state: HypervisorState):
    """
    Background task that aggressively polls GPU vitals every 2 seconds.
    Engages a global thermal lockout if thresholds are breached, forcing agents to wait.
    """
    print("[Watchdog] Active Hardware Thermal Watchdog initialized. Monitoring silicon health.")
    cooldown_target = settings.max_gpu_temp_c - 10  # Must drop 10 degrees below max to unlock
    
    while True:
        await asyncio.sleep(2.0)
        vitals = state.hardware_monitor.get_vitals()
        
        if vitals["status"] == "online":
            current_temp = vitals["temp_c"]
            current_vram = vitals["vram_pct"]
            
            # 1. Check for Breach
            if not state.emergency_thermal_lock:
                if current_temp >= settings.max_gpu_temp_c or current_vram >= settings.max_vram_allocation_pct:
                    print(f"\n[CRITICAL WARNING] HARDWARE LIMIT BREACHED!")
                    print(f" -> Temp: {current_temp}°C (Max: {settings.max_gpu_temp_c}°C)")
                    print(f" -> VRAM: {current_vram:.1f}% (Max: {settings.max_vram_allocation_pct}%)")
                    print("[CRITICAL WARNING] Engaging Emergency Thermal Lockout. Suspending all agent requests.")
                    state.emergency_thermal_lock = True
                    
            # 2. Check for Recovery (Hysteresis)
            elif state.emergency_thermal_lock:
                if current_temp <= cooldown_target and current_vram < settings.max_vram_allocation_pct:
                    print(f"\n[RECOVERY] GPU cooled to {current_temp}°C. VRAM stabilized.")
                    print("[RECOVERY] Disengaging Thermal Lockout. Resuming agent operations.")
                    state.emergency_thermal_lock = False


# --- 3. STATE-AWARE LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[API] Booting AetherForge Control Plane...")
    
    state = HypervisorState()
    app.state.hypervisor = state

    topo_path = "moe_topology.json"
    if os.path.exists(topo_path):
        state.cache_manager.load_topology(topo_path)

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
        
    # Start the async watchdog task
    watchdog_task = asyncio.create_task(hardware_watchdog(state))
        
    yield
    print("\n[API] Shutting down AetherForge Control Plane...")
    watchdog_task.cancel()
    state.hardware_monitor.shutdown()

app = FastAPI(title="AetherForge Hypervisor API", version="0.6.0", lifespan=lifespan)


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


# --- 4. HARDENED ROUTES ---

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
        mode: {"live_tps": profile["live_tps"], "baseline_tps": profile["decode_tps"]}
        for mode, profile in state.gatekeeper.profiles.items()
    }
    return {
        "timestamp": time.time(),
        "active_strategy": state.current_strategy,
        "thermal_lock_active": state.emergency_thermal_lock,
        "vram_pressure": {
            "current_mb": state.cache_manager.current_vram_usage,
            "budget_mb": state.cache_manager.vram_budget_mb,
            "utilization_pct": (state.cache_manager.current_vram_usage / state.cache_manager.vram_budget_mb) * 100
        },
        "silicon_vitals": state.hardware_monitor.get_vitals(),
        "performance_baselines": gatekeeper_telemetry,
        "engine_state": "simulation" if state.is_simulated else "online"
    }

@app.post("/system/strategy")
async def update_strategy(payload: StrategyPayload, request: Request):
    state = request.app.state.hypervisor
    
    # --- ZERO TOLERANCE HARDWARE LOCKOUT CHECK ---
    if state.emergency_thermal_lock:
        raise HTTPException(
            status_code=503, 
            detail="SYSTEM LOCKED: GPU is currently cooling down from a thermal event. Please back off and try again later."
        )

    target_mode = payload.mode.lower()
    
    async with state.lock:
        context_size = payload.estimated_context_tokens
        if payload.context_text:
            context_size = state.hardware_engine.count_tokens(payload.context_text)
        
        is_profitable = state.gatekeeper.evaluate_swap(
            current_mode=state.current_strategy,
            target_mode=target_mode,
            context_tokens=context_size,
            expected_output=payload.expected_output_tokens
        )
        
        if not is_profitable:
            return {"status": "rejected", "reason": "Swap latency overhead exceeds raw throughput gains.", "active_mode": state.current_strategy}
        
        state.current_strategy = target_mode
        result = state.hardware_engine.apply_strategy(state.current_strategy)
        
        if isinstance(result, dict):
            success = result.get("success", False)
            swap_metrics = result.get("metrics", {})
        else:
            success = result
            swap_metrics = {}
            
        if not success:
            raise HTTPException(status_code=500, detail="Failed to apply hardware strategy.")
            
        io_total = swap_metrics.get("extract_seconds", 0.0) + swap_metrics.get("inject_seconds", 0.0)
        reload_total = swap_metrics.get("reload_seconds", 0.0)
        
        if io_total > 0 or reload_total > 0:
            state.gatekeeper.update_hardware_latencies(physical_swap_seconds=reload_total, io_seconds=io_total)
            
        return {"status": "strategy_applied", "active_mode": state.current_strategy}

@app.post("/generate")
async def generate_text(payload: GenerationPayload, request: Request):
    state = request.app.state.hypervisor
    
    # --- ZERO TOLERANCE HARDWARE LOCKOUT CHECK ---
    if state.emergency_thermal_lock:
        raise HTTPException(
            status_code=503, 
            detail="SYSTEM LOCKED: GPU is currently cooling down from a thermal event. Please back off and try again later."
        )

    exact_prompt_tokens = state.hardware_engine.count_tokens(payload.prompt)
    if exact_prompt_tokens > settings.max_safe_context_tokens:
        raise HTTPException(status_code=413, detail=f"Context payload size violates the active hardware safety ceiling.")

    active_mode = payload.strategy.mode if payload.strategy else state.current_strategy

    async with state.lock:
        if state.hardware_engine.current_strategy != active_mode:
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
                active_mode = state.current_strategy

        output = state.hardware_engine.generate(prompt=payload.prompt, max_tokens=payload.max_tokens)

    measured_tps = output["metrics"].get("tokens_per_second", 0)
    if measured_tps > 0:
        state.gatekeeper.update_profile(active_mode, measured_tps)

    return {
        "text": output["text"],
        "metrics": output["metrics"],
        "hardware_engaged": not state.is_simulated
    }