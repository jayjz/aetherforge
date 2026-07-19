import time
import requests
import sys
from statistics import mean

API_URL = "http://localhost:8000"
TEST_ITERATIONS = 3  # Averages out system jitter

def print_vitals(stage: str):
    """Polls the AetherForge control plane for live hardware metrics."""
    try:
        res = requests.get(f"{API_URL}/system/metrics", timeout=5)
        if res.status_code == 200:
            metrics = res.json()
            temp = metrics['silicon_vitals']['temp_c']
            vram = metrics['vram_pressure']['utilization_pct']
            print(f"   [Vitals - {stage}] GPU Temp: {temp}°C | VRAM Usage: {vram:.1f}%")
    except requests.exceptions.RequestException:
        pass # Fail silently on vitals check to not interrupt the benchmark

def check_safety_interlocks():
    print("[*] Polling AetherForge control plane...")
    try:
        res = requests.get(f"{API_URL}/system/metrics", timeout=5)
        res.raise_for_status()
        metrics = res.json()
        
        if metrics.get("thermal_lock_active"):
            print("\n[!] FATAL: AetherForge is currently in Emergency Thermal Lockout.")
            print("[!] Did you forget to remove the 90°C spoof in server.py?")
            sys.exit(1)
            
        print(f"[+] API Online. Engine State: {metrics['engine_state']}")
    except requests.exceptions.RequestException as e:
        print(f"\n[!] FATAL: Control plane unreachable. Is Docker running? Error: {e}")
        sys.exit(1)

def generate_agent_payload(token_count: int) -> str:
    print(f"[*] Synthesizing ~{token_count}-token agent payload...")
    base_string = "This is a simulated memory block from Hermes agent history. "
    multiplier = (token_count * 4) // len(base_string)
    context = base_string * multiplier
    return context + "\n\nBased on the above history, what is the next optimal action?"

def run_benchmark(name: str, payload: str, strategy: str) -> float:
    print(f"\n{'-'*60}")
    print(f"[*] RUNNING TEST: {name} ({TEST_ITERATIONS} Iterations)")
    print(f"[*] Strategy Mode: {strategy}")
    print(f"{'-'*60}")
    
    wall_times = []
    
    for i in range(TEST_ITERATIONS):
        print(f"\n   --- Iteration {i+1}/{TEST_ITERATIONS} ---")
        print_vitals("Pre-Run")
        
        try:
            # Force the strategy
            requests.post(f"{API_URL}/system/strategy", json={
                "mode": strategy,
                "estimated_context_tokens": len(payload) // 4,
                "expected_output_tokens": 50
            }, timeout=10)

            gen_data = {
                "prompt": payload,
                "max_tokens": 50,
                "strategy": {"mode": strategy}
            }

            start_time = time.perf_counter()
            response = requests.post(f"{API_URL}/generate", json=gen_data, timeout=120)
            end_time = time.perf_counter()
            
            if response.status_code == 503:
                print("\n[!] TEST FAILED: GPU hit thermal lockout mid-generation!")
                sys.exit(1)
            response.raise_for_status()
            
            wall_clock_time = end_time - start_time
            wall_times.append(wall_clock_time)
            
            data = response.json()
            tps = data.get("metrics", {}).get("tokens_per_second", 0)
            
            print(f"   Time: {wall_clock_time:.2f}s | TPS: {tps:.2f}")
            print_vitals("Post-Run")
            
            if i < TEST_ITERATIONS - 1:
                print("   [+] Cooling down for 8 seconds...")
                time.sleep(8)
                
        except requests.exceptions.RequestException as e:
            print(f"\n[!] TEST FAILED: Network/API Error during generation: {e}")
            sys.exit(1)
            
    avg_time = mean(wall_times)
    print(f"\n[+] AVERAGE WALL-CLOCK TIME: {avg_time:.2f} seconds")
    return avg_time

if __name__ == "__main__":
    print("=== AETHERFORGE 'AGENT TAX' BENCHMARK SUITE ===")
    check_safety_interlocks()
    
    # LOWERED: Started at 2000 to safely test 8GB boundaries without immediate OOM.
    agent_prompt = generate_agent_payload(2000)
    
    time_control = run_benchmark("Static Baseline (VRAM Bound)", agent_prompt, "high_fidelity")
    
    print("\n[*] Inter-test cooling period (10 seconds)...")
    time.sleep(10)
    
    time_variable = run_benchmark("AetherForge Dynamic Swap (RAM -> VRAM)", agent_prompt, "aggressive_quant")
    
    print("\n=== FINAL ANALYSIS ===")
    if time_variable < time_control:
        diff = time_control - time_variable
        print(f"[WIN] AetherForge was {diff:.2f} seconds FASTER on average.")
    else:
        diff = time_variable - time_control
        print(f"[LOSS] AetherForge was {diff:.2f} seconds SLOWER on average.")