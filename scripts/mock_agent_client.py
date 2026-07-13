import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def check_cache(client: httpx.AsyncClient):
    """Polls the hypervisor to see how constrained VRAM is."""
    print("\n[Agent] Checking hypervisor cache status...")
    response = await client.get(f"{BASE_URL}/system/cache")
    if response.status_code == 200:
        data = response.json()
        print(f"  -> Status: {data['status'].upper()}")
        print(f"  -> VRAM Usage: {data['current_vram_usage_mb']:.0f} / {data['vram_budget_mb']:.0f} MB")
        print(f"  -> Active Experts in VRAM: {len(data['active_experts_in_vram'])}")
    else:
        print(f"  -> Error: {response.status_code}")

async def set_strategy(client: httpx.AsyncClient, mode: str):
    """Commands the hypervisor to change its memory management strategy."""
    print(f"\n[Agent] Injecting new strategy: {mode.upper()}")
    payload = {"mode": mode, "priority_layers": [0, 15, 31]} # Example priority layers
    response = await client.post(f"{BASE_URL}/system/strategy", json=payload)
    if response.status_code == 200:
        print("  -> Strategy accepted by AetherForge.")

async def generate_text(client: httpx.AsyncClient, prompt: str, mode: str):
    """Triggers generation while passing down a context-specific strategy."""
    print(f"\n[Agent] Sending inference request: '{prompt[:30]}...'")
    
    payload = {
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2,
        "strategy": {"mode": mode}
    }
    
    # We await the generation. In the real system, you might stream this.
    response = await client.post(f"{BASE_URL}/generate", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"  -> Generation Complete. Tokens: {data.get('tokens_generated')}")
        print(f"  -> Response text: {data.get('text')}")

async def run_simulation():
    print("=== OpenClaw Mock Agent Booting ===")
    
    async with httpx.AsyncClient() as client:
        # 1. Check initial state
        await check_cache(client)
        
        # 2. Agent decides it has a complex coding task. Demands High Fidelity.
        await set_strategy(client, "high_fidelity")
        
        # 3. Fire off the request
        await generate_text(client, "Write a complex Python script to manage a multidimensional cache...", "high_fidelity")
        
        # 4. Check state mid-workflow
        await check_cache(client)
        
        # 5. Agent switches context to summarizing a casual chat. Demands Aggressive Quantization to save power/VRAM.
        await set_strategy(client, "aggressive_quant")
        await generate_text(client, "Summarize the last 10 messages from my Telegram group.", "aggressive_quant")

if __name__ == "__main__":
    asyncio.run(run_simulation())