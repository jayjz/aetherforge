# AetherForge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: v0.4.0](https://img.shields.io/badge/status-v0.4.0-green.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

**Agent-aware memory hypervisor for local MoE inference on consumer GPUs.**

AetherForge sits between autonomous agents and llama.cpp. It lets agents request different VRAM strategies mid-session while preserving the KV-cache, so long-context multi-step workflows no longer pay a full prefill penalty on every strategy change.

**Current production path (v0.4.0):** Controlled Fast-Swap (model reload with different `n_gpu_layers`) + KV-cache serialization + Economic Gatekeeper + OpenAI-compatible tool schema.  
True in-memory expert routing remains on the research track.

---

## Why this exists

On 8–16 GB cards, static layer pinning forces a painful trade-off: either you keep everything in VRAM (and OOM on long context) or you run a heavy hybrid and accept slow generation. Agents that switch between light routing and heavy coding amplify the problem.

AetherForge makes the strategy change itself cheap enough that agents can request it. The Economic Gatekeeper rejects swaps whose latency cost exceeds the expected throughput gain.

---

## Features (what actually works today)

- **Fast-Swap with KV survival** — Tear down and reload with a different layer count; context is restored from system RAM.
- **Economic Gatekeeper** — Deterministic ROI check before any hardware change.
- **Configuration-driven** — All paths, layer counts, TPS profiles, and heuristics live in `.env`. No magic numbers in source.
- **Agent-discoverable** — `/system/tools` exports a live OpenAI function schema generated from the same Pydantic models the API uses.
- **Fail-fast startup** — Missing model path aborts with a clear error instead of silent simulation mode.

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA toolkit matching your GPU (for accelerated `llama-cpp-python`)
- A GGUF model (defaults assume DeepSeek-Coder-V2-Lite-Instruct Q4_K_M)

```bash
git clone https://github.com/jayjz/aetherforge.git
cd aetherforge
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Important:** `llama-cpp-python` must be built with CUDA support.  
> Example (Linux):  
> `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`

### Configuration

```bash
cp .env.example .env
# Edit .env — set MODEL_PATH and calibrate layer counts / TPS for your hardware
```

Defaults are calibrated for an **RTX 4060 8 GB + DeepSeek-Lite**. Change them for other cards or models so the Gatekeeper math stays accurate.

### Run

```bash
uvicorn src.server:app --host 127.0.0.1 --port 8000
```

### Example agent call

```bash
curl -X POST http://127.0.0.1:8000/system/strategy \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "high_fidelity",
    "context_text": "long conversation history here...",
    "expected_output_tokens": 800
  }'
```

Or let an agent discover the tool via `GET /system/tools`.

---

## Architecture (current)

```
Agent ──► FastAPI Control Plane
              │
              ├─ Economic Gatekeeper (ROI decision)
              ├─ Cache Manager (topology + state)
              └─ AetherEngine
                    │
                    └─ llama.cpp (Fast-Swap + KV save/load)
```

The production path intentionally uses a full model reload + state restore. This is reliable and works with stock `llama-cpp-python`. Research into true runtime expert movement lives on separate branches.

---

## Performance reference

Measured on RTX 4060 8 GB, DeepSeek-Coder-V2-Lite-Instruct Q4_K_M:

| Strategy          | Layers in VRAM | Approx. t/s | Typical use                  |
|-------------------|----------------|-------------|------------------------------|
| high_fidelity     | 15             | ~23.7       | Coding, multi-step reasoning |
| balanced          | 10             | ~11.1       | Normal agent dialog          |
| aggressive_quant  | 2              | ~12.1       | Routing / light tasks        |

Swap cost is typically 5–8 s (dominated by reload). Because the KV-cache survives, the prefill penalty on the next generation is largely eliminated.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the current phased plan.

- **v0.4.0 (now)** — Centralized config, validation, Fail-fast, tool schema, Fast-Swap + KV survival.
- Next — Mock Engine for CI, better CUDA onboarding, real LangGraph/n8n examples.
- Research track — True dynamic expert movement (separate from the production path).
### Live Telemetry & Learning
AetherForge does not rely blindly on static configuration. The `EconomicGatekeeper` uses an Exponential Moving Average (EMA) to track your actual generation speeds. If your GPU thermally throttles, the Gatekeeper automatically adjusts its ROI math to account for the degraded performance. 
*   **Smoothing (`TELEMETRY_ALPHA`):** Defaults to 0.3 (30% new measurement, 70% historical) to ignore transient OS spikes.
*   **Clamping:** Hard floors and ceilings (`TPS_MIN_CLAMP` / `TPS_MAX_CLAMP`) prevent a single crashed generation from permanently corrupting the routing logic.
---

## Project guidelines

See [PROJECTGUIDELINES.md](PROJECTGUIDELINES.md) for architecture principles, branching rules, decision log, and the “stranger test” definition of done.

---

## Contributing

Issues and PRs are welcome. Keep `main` always runnable. Prefer short-lived feature branches and conventional commits. Research experiments should stay on `research/*` branches until stable.
```

---