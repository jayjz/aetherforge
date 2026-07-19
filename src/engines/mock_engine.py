"""
AetherForge Hardware Emulator (The Mock Muscle)
==============================================
A purely software-driven emulation of the AetherEngine.
Simulates Fast-Swap latencies and generation speeds (TPS) based on 
the configuration file, allowing full Control Plane testing without a GPU.
"""

import time
from typing import Dict, Any
from src.config import settings
from src.engines.base import BaseAetherEngine

class MockAetherEngine(BaseAetherEngine):
    def __init__(self, model_path: str, vram_budget_mb: float = 8000, n_ctx: int = 4096):
        self.model_path = model_path
        self.vram_budget_mb = vram_budget_mb
        self.n_ctx = n_ctx
        self.current_strategy = "balanced"
        
        # Map configuration TPS to simulate accurate generation times
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
        """Simulates the physical VRAM Fast-Swap penalty and returns telemetry metrics."""
        # The expected return structure
        metrics = {"extract_seconds": 0.0, "reload_seconds": 0.0, "inject_seconds": 0.0}
        
        if mode not in self.tps_map:
            print(f"[MockEngine] Invalid strategy '{mode}'. Defaulting to balanced.")
            mode = "balanced"
            
        if mode == self.current_strategy:
            return {"success": True, "metrics": metrics} # No-op
            
        print(f"\n[MockEngine] SIMULATED HARDWARE OVERRIDE INITIATED.")
        print(f" -> Shifting from '{self.current_strategy}' to '{mode}'.")
        
        # 1. Simulate Extraction
        mock_extract = settings.state_io_base_seconds * 0.5
        time.sleep(mock_extract)
        metrics["extract_seconds"] = mock_extract
        
        # 2. Simulate Reload
        mock_reload = settings.swap_penalty_seconds
        print(f" -> [Simulation] Emulating {mock_reload}s physical model reload...")
        time.sleep(mock_reload)
        metrics["reload_seconds"] = mock_reload
        
        # 3. Simulate Injection
        mock_inject = settings.state_io_base_seconds * 0.5
        time.sleep(mock_inject)
        metrics["inject_seconds"] = mock_inject
        
        self.current_strategy = mode
        print(f"[MockEngine] Simulated Fast-Swap Protocol Complete.")
        
        return {"success": True, "metrics": metrics}

    def generate(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Simulates token generation constrained by the active strategy's TPS limit."""
        print(f"\n[MockEngine] Commencing simulated generation (Mode: {self.current_strategy.upper()})...")
        
        target_tps = self.tps_map[self.current_strategy]
        simulated_time = max_tokens / target_tps
        
        print(f" -> [Simulation] Emulating {max_tokens} tokens at {target_tps} t/s...")
        time.sleep(simulated_time)
        
        return {
            "text": f"[MOCK GENERATION] AetherForge simulated {max_tokens} tokens successfully under {self.current_strategy} constraints.",
            "metrics": {
                "tokens_generated": max_tokens,
                "time_seconds": simulated_time,
                "tokens_per_second": target_tps,
                "active_strategy": self.current_strategy
            }
        }