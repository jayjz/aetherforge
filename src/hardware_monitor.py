"""
AetherForge Hardware Monitor
============================
Reads physical silicon vitals via NVML. 
Gracefully degrades to headless simulation if no NVIDIA driver is present.
Injects probabilistic hardware chaos when running in simulation mode.
"""

import os
import random
from typing import Dict, Union

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

class HardwareMonitor:
    def __init__(self):
        self.active: bool = False
        self.handle = None
        
        # Simulation state for headless testing
        self._simulated_cooldown_ticks: int = 0
        
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.active = True
                print("[HardwareMonitor] NVML bindings established. Silicon vitals online.")
            except Exception as e:
                print(f"[HardwareMonitor] NVML init failed ({e}). Operating blind.")
        else:
            print("[HardwareMonitor] pynvml not detected. Operating in headless simulation.")

    def get_vitals(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns physical thermals and VRAM usage. 
        If offline, simulates a sticky hardware state machine for testing API circuit breakers.
        """
        if not self.active:
            # 1. Decay active thermal events
            if self._simulated_cooldown_ticks > 0:
                self._simulated_cooldown_ticks -= 1
                # Hold at 92°C to enforce the 503 lockout without staying at the absolute peak
                return {"temp_c": 92, "vram_pct": 98.0, "status": "simulated"}

            # 2. Roll for new thermal event only if explicitly opted-in via environment
            chaos_enabled = os.getenv("AETHER_CHAOS", "false").lower() == "true"
            if chaos_enabled and random.random() < 0.10:
                print("\n[HardwareMonitor] SIMULATED CHAOS: Thermal event initiated! Holding for 12 seconds...")
                self._simulated_cooldown_ticks = 5  # 5 ticks * 2s polling interval = ~10s of sustained heat
                return {"temp_c": 95, "vram_pct": 99.0, "status": "simulated"}

            # 3. Baseline idle state
            return {"temp_c": 45, "vram_pct": 40.0, "status": "simulated"}
        
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            vram_pct = (mem_info.used / mem_info.total) * 100.0
            return {"temp_c": temp, "vram_pct": vram_pct, "status": "online"}
        except Exception as e:
            print(f"[HardwareMonitor] Polling error: {e}")
            return {"temp_c": 0, "vram_pct": 0.0, "status": "error"}
            
    def shutdown(self) -> None:
        if self.active:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass