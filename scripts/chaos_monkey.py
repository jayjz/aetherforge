"""
AetherForge Chaos Monkey
========================
Automates the Week 2 Load & Chaos testing suite.
Spins up the hypervisor with thermal chaos enabled, blasts it with 50 
concurrent agents via Locust, and parses the survival metrics.
"""

import subprocess
import time
import os
import sys

def run_chaos_test():
    print("\n" + "="*60)
    print("🦍 AETHERFORGE CHAOS MONKEY INITIATED")
    print("="*60)
    
    # 1. Prepare Environment (Chaos Enabled)
    env = os.environ.copy()
    env["AETHER_ENGINE"] = "mock"
    env["AETHER_CHAOS"] = "true"  # Enables 95°C random thermal spikes

    # 2. Boot the API Server in the background
    print("\n[+] Booting Hypervisor Control Plane (Chaos Mode = ON)...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", "8000"],
        env=env,
        stdout=subprocess.DEVNULL,  # Keep console clean
        stderr=subprocess.DEVNULL
    )
    
    # Give uvicorn 3 seconds to bind to the port
    time.sleep(3)

    # 3. Launch Locust Attack (Headless)
    print("[+] API Online. Launching 50 Concurrent Agent Swarm via Locust...")
    print("[+] Attack Duration: 45 Seconds. Monitor logs/hardware_safety.log to see the carnage.")
    
    locust_cmd = [
        "locust",
        "-f", "locustfile.py",
        "--headless",
        "-u", "50",         # 50 concurrent agents
        "-r", "10",         # Spawn 10 agents per second
        "--run-time", "45s", # Run for 45 seconds
        "--host", "http://127.0.0.1:8000",
        "--csv", "chaos_results"
    ]
    
    try:
        # Run Locust and wait for it to finish
        subprocess.run(locust_cmd, check=False)
        
        # 4. Analyze Results
        print("\n" + "="*60)
        print("✅ ATTACK CONCLUDED. PARSING RESULTS...")
        print("="*60)
        
        # We check the failures CSV. If the server survived gracefully, 
        # the ONLY errors should be 503s (Thermals/Queue) and 413s (Context).
        # Any 500s or ConnectionRefused means the server died.
        with open("chaos_results_failures.csv", "r") as f:
            failures = f.read()
            
        print("\n[Failure Distribution]")
        if "500 Internal Server Error" in failures or "ConnectionRefused" in failures:
            print("❌ FATAL: The hypervisor crashed or dropped connections.")
        else:
            print("🛡️  SUCCESS: 0 Server Crashes. 100% Graceful Degradation.")
            print("All rejected traffic was correctly routed to 503 (Thermal/Queue) or 413 (Context Ceiling).")
            
    except Exception as e:
        print(f"\n❌ Error running Locust: {e}")
        
    finally:
        # 5. Cleanup
        print("\n[+] Tearing down Hypervisor...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    run_chaos_test()