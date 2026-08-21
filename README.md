# AetherForge

**Hardware-Aware Safety Control Plane for Autonomous Agents on Consumer GPUs.**

AetherForge is an L7 API gateway and negotiation layer that sits between autonomous AI agents and local inference engines. It is designed to prevent rogue or looping agents from melting consumer GPUs (e.g., RTX 4060 8GB), causing CUDA Out-Of-Memory (OOM) crashes, or freezing event loops.

### The Agent-Infrastructure Contract
Pure inference engines (llama.cpp, vLLM) optimize for throughput under static memory budgets. They treat the caller as a dumb pipe. AetherForge treats the agent as a first-class client by enforcing a strict negotiation contract:

1. **Strategy Discovery:** Agents can query `/system/tools` and `/system/metrics` to inspect hardware health and available VRAM strategies.
2. **Strategy Request:** Agents proactively request a VRAM strategy change (e.g., `high_fidelity` vs `balanced`) with estimated token requirements *before* generation. The Economic Gatekeeper evaluates the ROI of the latency penalty.
3. **Explicit Economic & Thermal Rejection:** The control plane intercepts dangerous payloads and returns structured, machine-readable HTTP `413` (Context Ceiling) or `503` (Thermal Lock / Queue Saturated) JSON responses with explicit `Retry-After` headers.
4. **Durable Audit Trail:** (In Progress) Every Gatekeeper intervention emits structured JSON logs, providing an immutable record of *why* the control plane intervened.

### Architecture Status
> **Verified Today (Wedge A):** The safety control plane, Economic Gatekeeper, concurrent queue admission, and structured 503/413 circuit breakers are verified under `AETHER_ENGINE=mock`. This allows agent resilience testing without requiring dedicated CUDA hardware.
>
> **Experimental (Wedge B):** Physical Fast-Swap with reliable VRAM teardown and KV-cache survival. These components have been quarantined to `src/research_archive/` and are not on the production hot-path.

### Quick Start (Headless Simulation)
You can run the AetherForge control plane in mock mode to test agent resilience and API contracts without an NVIDIA GPU.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Boot the API with the Mock Engine
AETHER_ENGINE=mock uvicorn src.server:app --host 127.0.0.1 --port 8000

# 3. (Optional) Run the Chaos Monkey load suite to trigger thermal/queue limits
AETHER_ENGINE=mock AETHER_CHAOS=true python scripts/chaos_monkey.py