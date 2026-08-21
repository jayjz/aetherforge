# ADR 002: Agent-Infrastructure Negotiation as Core Product Surface

## Status
Accepted

## Context
Pure inference engines (llama.cpp, vLLM, KTransformers) optimize for throughput and latency under a static memory budget. They treat the caller as a dumb pipe. When autonomous agents (e.g., LangChain, AutoGen) interact with these engines, they blindly send massive context windows (e.g., 50k tokens) to constrained hardware (e.g., 8GB GPUs), causing unrecoverable OOM crashes, thermal throttling, or event-loop freezes. 

Existing literature and tools lack a backpressure and economic awareness layer that treats the external agent as a first-class client.

## Decision
AetherForge will operate strictly as an **L7 Control Plane** wrapping the inference engine. It will not compete on raw matrix multiplication speed. Instead, it enforces a strict **Agent-Infrastructure Negotiation Contract** guaranteeing four capabilities:

1. **Strategy Discovery**: Agents can query `/system/tools` and `/system/metrics` to understand available VRAM strategies and current hardware health.
2. **Strategy Request**: Agents can explicitly request a strategy change (e.g., `high_fidelity`) with estimated context/output tokens *before* generation.
3. **Explicit Economic/Thermal Rejection**: The API will return structured, machine-readable HTTP 413 (Context Ceiling) or 503 (Thermal Lock / Queue Saturated) responses with explicit `Retry-After` headers and JSON error codes.
4. **Durable Audit Trail**: Every Gatekeeper intervention is logged as structured JSON, allowing external systems to query exactly *why* a request was rejected (e.g., "ROI negative: io_overhead_exceeds_generation_gain").

## Consequences
- **Positive**: Agents become resilient. They can catch a 503, sleep for the `Retry-After` duration, truncate their context on a 413, and retry successfully.
- **Negative**: We explicitly deprioritize raw inference speed optimizations. Experimental VRAM teardown/fast-swap logic (`cache_manager.py`, `tensor_bridge.py`) is moved to `src/research_archive/` and excluded from the primary production hot-path.
