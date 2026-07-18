"""
AetherForge Hardware Monitor
============================
Reads physical silicon vitals via NVML. 
Gracefully degrades to headless simulation if no NVIDIA driver is present.
"""

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

class HardwareMonitor:
    def __init__(self):
        self.active = False
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                # Default to GPU 0 (your RTX 4060)
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.active = True
                print("[HardwareMonitor] NVML bindings established. Silicon vitals online.")
            except Exception as e:
                print(f"[HardwareMonitor] NVML init failed ({e}). Operating blind.")
        else:
            print("[HardwareMonitor] pynvml not detected. Operating in headless simulation.")

    def get_vitals(self) -> dict:
        """Returns physical thermals and VRAM usage. Returns zeroes if offline."""
        if not self.active:
            return {"temp_c": 0, "vram_pct": 0.0, "status": "simulated"}
        
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            vram_pct = (mem_info.used / mem_info.total) * 100.0
            return {"temp_c": temp, "vram_pct": vram_pct, "status": "online"}
        except Exception as e:
            print(f"[HardwareMonitor] Polling error: {e}")
            return {"temp_c": 0, "vram_pct": 0.0, "status": "error"}
            
    def shutdown(self):
        if self.active:
            try:
                pynvml.nvmlShutdown()
            except:
                pass