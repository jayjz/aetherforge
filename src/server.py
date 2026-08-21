import os
import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.config import settings
from src.engines import create_engine
from src.hardware_monitor import HardwareMonitor

from src.logger import setup_logging, get_logger, get_audit_logger

setup_logging()
api_logger = get_logger("api")
gatekeeper_logger = get_logger("gatekeeper")
watchdog_logger = get_logger("watchdog")
audit_logger = get_audit_logger() # <-- NEW

# --- 1. HARDENED STATE CONTAINER ---
class HypervisorState:
    def __init__(self):
        self.gatekeeper = EconomicGatekeeper()
        self.hardware_monitor = HardwareMonitor()
        self.hardware_engine = None
        self.current_strategy = "balanced"
        self.is_simulated = True 
        
        # Atomic concurrency control wired directly to configuration
        self.semaphore = asyncio.Semaphore(settings.max_queue_depth)
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
        self.profiles[mode]["live_tps"] = max(settings.tps_min_clamp, min(settings.tps_max_clamp, raw_ema))

    def evaluate_swap(self, current_mode: str, target_mode: str, context_tokens: int, expected_output: int) -> bool:
        if current_mode == target_mode:
            return False 

        if context_tokens > settings.max_safe_context_tokens:
            gatekeeper_logger.warning(f"Swap Rejected: Context size ({context_tokens}) exceeds ceiling.")
            return False

        current_tps = self.profiles.get(current_mode, self.profiles["balanced"])["live_tps"]
        target_tps = self.profiles.get(target_mode, self.profiles["balanced"])["live_tps"]
        
        time_to_stay = expected_output / current_tps
        state_io_overhead = self.live_io_penalty + (context_tokens * settings.state_io_per_token_seconds)
        target_generation_time = expected_output / target_tps
        
        if (self.live_swap_penalty + state_io_overhead + target_generation_time) >= time_to_stay:
            return False
        return True

# --- 2. ASYNCHRONOUS HARDWARE WATCHDOG ---
async def hardware_watchdog(state: HypervisorState):
    watchdog_logger.info("Active Hardware Thermal Watchdog initialized. Monitoring silicon health.")
    cooldown_target = settings.max_gpu_temp_c - 10 
    
    while True:
        await asyncio.sleep(2.0)
        vitals = state.hardware_monitor.get_vitals()
        
        if vitals["status"] in ["online", "simulated"]:
            current_temp = vitals["temp_c"]
            current_vram = vitals["vram_pct"]
            
            if not state.emergency_thermal_lock:
                if current_temp >= settings.max_gpu_temp_c or current_vram >= settings.max_vram_allocation_pct:
                    watchdog_logger.critical(f"HARDWARE LIMIT BREACHED! Temp: {current_temp}°C | VRAM: {current_vram:.1f}%")
                    state.emergency_thermal_lock = True
            elif state.emergency_thermal_lock:
                if current_temp <= cooldown_target and current_vram < settings.max_vram_allocation_pct:
                    watchdog_logger.info(f"RECOVERY: GPU cooled to {current_temp}°C. VRAM stabilized.")
                    state.emergency_thermal_lock = False

# --- 3. STATE-AWARE LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    api_logger.info("Booting AetherForge Control Plane...")
    
    state = HypervisorState()
    app.state.hypervisor = state

    target_engine = settings.aether_engine.lower()
    if target_engine == "auto":
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
        api_logger.critical(f"Fatal Engine Factory crash: {e}")
        raise RuntimeError(f"Hypervisor engine initialization failed: {e}")
        
    watchdog_task = asyncio.create_task(hardware_watchdog(state))
    yield
    api_logger.info("Shutting down AetherForge Control Plane...")
    watchdog_task.cancel()
    state.hardware_monitor.shutdown()

app = FastAPI(title="AetherForge Hypervisor API", version="0.7.0", lifespan=lifespan)

# --- PYDANTIC SCHEMAS ---
class StrategyPayload(BaseModel):
    mode: str = Field(..., description="The VRAM strategy: 'high_fidelity', 'balanced', or 'aggressive_quant'.")
    estimated_context_tokens: Optional[int] = Field(500)
    expected_output_tokens: Optional[int] = Field(300)
    context_text: Optional[str] = Field(None)

class GenerationPayload(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    strategy: Optional[StrategyPayload] = None

# --- 4. HARDENED L7 ROUTES ---
@app.get("/system/metrics")
async def get_metrics(request: Request):
    state = request.app.state.hypervisor
    return {
        "timestamp": time.time(),
        "active_strategy": state.current_strategy,
        "thermal_lock_active": state.emergency_thermal_lock,
        "vram_pressure": {"utilization_pct": state.hardware_monitor.get_vitals().get("vram_pct", 0.0)},
        "silicon_vitals": state.hardware_monitor.get_vitals(),
        "performance_baselines": {m: {"live_tps": p["live_tps"]} for m, p in state.gatekeeper.profiles.items()},
        "engine_state": "simulation" if state.is_simulated else "online"
    }

@app.get("/system/tools")
async def get_tool_schema():
    schema = StrategyPayload.model_json_schema()
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        prop.pop("default", None)
    return {
        "type": "function",
        "function": {
            "name": "aetherforge_optimize_vram",
            "description": "Hypervisor control: Dynamically allocates physical VRAM layers based on task complexity.",
            "parameters": {"type": "object", "properties": schema.get("properties", {}), "required": schema.get("required", [])}
        }
    }

@app.post("/system/strategy")
async def update_strategy(payload: StrategyPayload, request: Request):
    state = request.app.state.hypervisor
    
    if state.emergency_thermal_lock:
        raise HTTPException(status_code=503, headers={"Retry-After": "10"}, detail={
            "error": "thermal_lock_active", "temp_c": state.hardware_monitor.get_vitals().get("temp_c", 0), "retry_after_seconds": 10
        })

    if state.semaphore.locked():
        raise HTTPException(status_code=503, headers={"Retry-After": "2"}, detail={
            "error": "queue_saturated", "queue_depth": settings.max_queue_depth, "retry_after_seconds": 2
        })

    await state.semaphore.acquire()
    try:
        target_mode = payload.mode.lower()
        context_size = state.hardware_engine.count_tokens(payload.context_text) if payload.context_text else payload.estimated_context_tokens
        
        if not state.gatekeeper.evaluate_swap(state.current_strategy, target_mode, context_size, payload.expected_output_tokens):
            gatekeeper_logger.info(f"Swap Rejected: {state.current_strategy} -> {target_mode} (Unprofitable ROI)")
            
            # --- NEW: Emit structured JSON audit ---
            audit_logger.info("Gatekeeper Intervention: Strategy Swap Rejected", extra={
                "details": {
                    "current_mode": state.current_strategy,
                    "target_mode": target_mode,
                    "context_tokens": context_size,
                    "expected_output": payload.expected_output_tokens,
                    "reason": "roi_negative"
                }
            })
            # ---------------------------------------
            
            return {"status": "rejected", "error": "roi_negative", "active_mode": state.current_strategy}
        
        result = await asyncio.to_thread(state.hardware_engine.apply_strategy, target_mode)
        success = result.get("success", False) if isinstance(result, dict) else result
            
        if not success:
            raise HTTPException(status_code=500, detail={"error": "hardware_swap_failed"})
        
        state.current_strategy = target_mode
        return {"status": "strategy_applied", "active_mode": state.current_strategy}
    finally:
        state.semaphore.release()

@app.post("/generate")
async def generate_text(payload: GenerationPayload, request: Request):
    state = request.app.state.hypervisor
    
    if state.emergency_thermal_lock:
        raise HTTPException(status_code=503, headers={"Retry-After": "10"}, detail={
            "error": "thermal_lock_active", "temp_c": state.hardware_monitor.get_vitals().get("temp_c", 0), "retry_after_seconds": 10
        })

    exact_prompt_tokens = state.hardware_engine.count_tokens(payload.prompt)
    if exact_prompt_tokens > settings.max_safe_context_tokens:
        raise HTTPException(status_code=413, detail={
            "error": "context_exceeded", "max_allowed": settings.max_safe_context_tokens, "attempted": exact_prompt_tokens, "action": "truncate_and_retry"
        })

    if state.semaphore.locked():
        raise HTTPException(status_code=503, headers={"Retry-After": "2"}, detail={
            "error": "queue_saturated", "queue_depth": settings.max_queue_depth, "retry_after_seconds": 2
        })

    await state.semaphore.acquire()
    try:
        active_mode = payload.strategy.mode if payload.strategy else state.current_strategy
        if state.hardware_engine.current_strategy != active_mode:
            if state.gatekeeper.evaluate_swap(state.current_strategy, active_mode, exact_prompt_tokens, payload.max_tokens):
                swap_res = await asyncio.to_thread(state.hardware_engine.apply_strategy, active_mode)
                if isinstance(swap_res, dict) and swap_res.get("success", False) or swap_res is True:
                    state.current_strategy = active_mode
            else:
                active_mode = state.current_strategy

        output = await asyncio.to_thread(state.hardware_engine.generate, prompt=payload.prompt, max_tokens=payload.max_tokens, temperature=payload.temperature)
        measured_tps = output.get("metrics", {}).get("tokens_per_second", 0)
        if measured_tps > 0:
            state.gatekeeper.update_profile(active_mode, measured_tps)
            
        return {"text": output.get("text", ""), "metrics": output.get("metrics", {}), "hardware_engaged": not state.is_simulated}
    finally:
        state.semaphore.release()