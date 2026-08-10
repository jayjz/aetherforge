"""
AetherForge Locust Swarm Definition
===================================
Simulates an aggressive multi-agent environment hitting the control plane.
Mixes standard generations, heavy 'Context Bombs', and VRAM swap requests.
"""

from locust import HttpUser, task, between

class AetherAgent(HttpUser):
    # Agents wait 1 to 3 seconds between actions
    wait_time = between(1.0, 3.0)

    @task(4)
    def standard_generation(self):
        """Simulates a normal, compliant agent workflow."""
        self.client.post("/generate", json={
            "prompt": "Standard log analysis request.",
            "max_tokens": 50,
            "strategy": {"mode": "balanced"}
        }, name="/generate (Standard)")

    @task(2)
    def strategy_thrash(self):
        """Simulates an agent attempting to aggressively re-allocate VRAM."""
        self.client.post("/system/strategy", json={
            "mode": "high_fidelity",
            "estimated_context_tokens": 100,
            "expected_output_tokens": 500
        }, name="/system/strategy (Swap)")

    @task(1)
    def context_bomb(self):
        """Simulates a runaway agent submitting a 20,000+ token payload (Triggers 413)."""
        heavy_prompt = "Agent memory dump " * 5000 
        self.client.post("/generate", json={
            "prompt": heavy_prompt,
            "max_tokens": 100
        }, name="/generate (Context Bomb)")