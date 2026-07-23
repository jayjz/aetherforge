"""
AetherForge Hardware Emulator (The Mock Muscle)
==============================================
A purely software-driven emulation of the AetherEngine.
Simulates Fast-Swap latencies, non-deterministic TPS jitter (±15%), 
and intermittent hardware failures to stress-test the Control Plane.
"""

import time
import random
from typing import Dict, Any
from src.config import settings
from src.engines.base import BaseAetherEngine

class MockAetherEngine(BaseAetherEngine):
    def __init__(self, model_path: str, vram_budget_mb: float = 8000, n_ctx: int = 4096):
        super().__init__()
        self.model_path = model_path
        self.vram_budget_mb = vram_budget_mb
        self.n_ctx = n_ctx
        self.current_strategy = "balanced"
        
        # Map configuration TPS to simulate baseline generation speeds
        self.tps_map = {
            "high_fidelity": settings.tps_high_fidelity,
            "balanced": settings.tps_balanced,
            "aggressive_quant": settings.tps_aggressive_quant
        }
        
        print(f"[MockEngine] Booting Software Emulation Core...")
        print(f"[MockEngine] Emulating model: {self.model_path}")
        print(f"[MockEngine] Mock Engine Online.")

    def count_tokens(self, text: str) -> int:
        """Approximates token count (roughly 4 chars per token)."""
        return max(1, len(text) // 4)

    def apply_strategy(self, mode: str) -> Dict[str, Any]:
        """Simulates physical Fast-Swap latencies with intermittent swap failures."""
        metrics = {"extract_seconds": 0.0, "reload_seconds": 0.0, "inject_seconds": 0.0}
        
        if mode not in self.tps_map:
            print(f"[MockEngine] Invalid strategy '{mode}'. Defaulting to balanced.")
            mode = "balanced"
            
        if mode == self.current_strategy:
            return {"success": True, "metrics": metrics}
            
        print(f"\n[MockEngine] SIMULATED HARDWARE OVERRIDE INITIATED.")
        print(f" -> Shifting from '{self.current_strategy}' to '{mode}'.")
        
        # 5% chance the simulated CUDA allocation fails
        if random.random() < 0.05:
            print(" -> [MockEngine] SIMULATED CHAOS: CUDA OutOfMemoryError during reload!")
            return {"success": False, "metrics": metrics}

        # 1. Simulate Extraction
        mock_extract = settings.state_io_base_seconds * 0.5
        time.sleep(mock_extract)
        metrics["extract_seconds"] = mock_extract
        
        # 2. Simulate Reload with slight latency jitter
        reload_jitter = settings.swap_penalty_seconds * random.uniform(-0.10, 0.10)
        mock_reload = max(0.1, settings.swap_penalty_seconds + reload_jitter)
        print(f" -> [Simulation] Emulating {mock_reload:.2f}s physical model reload...")
        time.sleep(mock_reload)
        metrics["reload_seconds"] = mock_reload
        
        # 3. Simulate Injection
        mock_inject = settings.state_io_base_seconds * 0.5
        time.sleep(mock_inject)
        metrics["inject_seconds"] = mock_inject
        
        self.current_strategy = mode
        print(f"[MockEngine] Simulated Fast-Swap Protocol Complete.")
        
        return {"success": True, "metrics": metrics}

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        """Simulates inference with ±15% TPS jitter to exercise the Gatekeeper EMA."""
        print(f"\n[MockEngine] Commencing simulated generation (Mode: {self.current_strategy.upper()}, Temp: {temperature:.2f})...")
        
        # Calculate TPS with ±15% non-deterministic jitter
        base_tps = self.tps_map.get(self.current_strategy, settings.tps_balanced)
        jitter = base_tps * random.uniform(-0.15, 0.15)
        actual_tps = max(settings.tps_min_clamp, base_tps + jitter)
        
        simulated_time = max_tokens / actual_tps
        
        # Cap sleep time to 1.5s so local development loop remains responsive
        time.sleep(min(simulated_time, 1.5))
        
        return {
            "text": f"[MOCK GENERATION] AetherForge simulated {max_tokens} tokens at {actual_tps:.2f} t/s (temp={temperature}).",
            "metrics": {
                "tokens_generated": max_tokens,
                "time_seconds": simulated_time,
                "tokens_per_second": actual_tps,
                "active_strategy": self.current_strategy
            }
        }