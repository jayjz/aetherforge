"""
AetherForge Telemetry Unit Test
===============================
Validates the EconomicGatekeeper's Exponential Moving Average (EMA) 
and clamping bounds without requiring hardware inference.
"""

from src.server import EconomicGatekeeper
from src.config import settings

def run_test():
    print("🧪 Running Gatekeeper Telemetry Tests...")
    gatekeeper = EconomicGatekeeper()
    mode = "high_fidelity"
    
    initial_tps = gatekeeper.profiles[mode]["live_tps"]
    print(f"Initial '{mode}' TPS: {initial_tps}")
    
    # Test 1: Normal standard deviation
    gatekeeper.update_profile(mode, 18.0)
    assert gatekeeper.profiles[mode]["live_tps"] < initial_tps, "EMA failed to adjust downward."
    
    # Test 2: Catastrophic throttle (Testing the floor clamp)
    print("\n[Simulating catastrophic thermal throttle down to 0.5 t/s]")
    gatekeeper.update_profile(mode, 0.5)
    gatekeeper.update_profile(mode, 0.5)
    current_tps = gatekeeper.profiles[mode]["live_tps"]
    assert current_tps >= settings.tps_min_clamp, f"Floor clamp failed! Dropped to {current_tps}"
    
    # Test 3: Unrealistic spike (Testing the ceiling clamp)
    print("\n[Simulating impossible 900 t/s spike]")
    gatekeeper.update_profile(mode, 900.0)
    current_tps = gatekeeper.profiles[mode]["live_tps"]
    assert current_tps <= settings.tps_max_clamp, f"Ceiling clamp failed! Spiked to {current_tps}"
    
    print("\n✅ All telemetry math and clamps passed successfully.")

if __name__ == "__main__":
    run_test()