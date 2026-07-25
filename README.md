# AetherForge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: control-plane / Wedge A](https://img.shields.io/badge/status-Wedge%20A%20safety%20layer-green.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

**Hardware-aware safety and control plane for local AI agents on consumer GPUs.**

AetherForge sits between autonomous agents and local inference. Agents can discover tools, request VRAM strategies, and generate text through a FastAPI control plane that enforces thermal/VRAM circuit breakers, economic swap decisions, and durable audit logs.

**What is verified today (Wedge A):** Mock engine path, Economic Gatekeeper, 503 thermal lock / 413 context limits, `/system/*` discovery APIs, rotating ops + safety logs, and a reference agent client — all under `AETHER_ENGINE=mock`.

**What is not verified:** Real Fast-Swap with reliable VRAM teardown and KV-cache survival on 8 GB cards (e.g. RTX 4060). That remains a gated research track (Wedge B). Do not treat it as production.

---

## Why this exists

Local agents on 8–16 GB GPUs fail in predictable ways: runaway generation, naive retries, context blow-ups, and strategy thrash that OOMs or thermally stresses the only card you have.

AetherForge’s job in the current sprint is not “move experts in VRAM.” It is:

1. Give agents a stable API to inspect health and request strategy changes.
2. Reject work that violates safety ceilings (temperature, VRAM pressure, context size).
3. Leave an audit trail when the control plane intervenes.
4. Allow full development and CI without CUDA via a Mock engine.

Long-term goal (research): agent-aware strategy switching with measured Fast-Swap / KV survival on consumer hardware. That is **not** the default claim of this README.

---

## Verified features (Wedge A)

| Feature | Status |
|--------|--------|
| FastAPI control plane (`/generate`, `/system/strategy`, `/system/metrics`, `/system/cache`, `/system/tools`) | Verified under Mock |
| Economic Gatekeeper (accept/reject swaps by ROI heuristics + EMA TPS) | Verified under Mock |
| Thermal / VRAM watchdog → `emergency_thermal_lock` → **503** | Verified (Mock chaos + forced lock tests) |
| Context ceiling → **413** | Verified in route logic / tests |
| OpenAI-style tool schema at `GET /system/tools` | Verified |
| Rotating logs: `logs/aetherforge.log`, `logs/hardware_safety.log` | Verified |
| Pytest suite (`tests/test_hypervisor.py`) | 3 tests, Mock-forced |
| Reference client `scripts/safe_agent.py` | Discovers tools, negotiates strategy, handles 503/413 |
| Lazy engine factory (Mock boots without importing CUDA bindings) | Required for headless / no-GPU machines |

---

## Explicit non-claims

- **Fast-Swap + KV survival on real GPUs** is experimental. `LlamaEngine` implements reload + `save_state` / `load_state` with best-effort teardown; VRAM barriers are **not** proven on 8 GB hardware.
- **True in-memory MoE expert movement** is not implemented. `ktransformers` support is a stub.
- Default docs and demos assume **Mock** unless you deliberately opt into `llama` after accepting hardware risk.

---

## Quick start (safe path — Mock)

```bash
git clone https://github.com/jayjz/aetherforge.git
cd aetherforge
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install pytest httpx   # for tests
```

### Configure for Mock

`.env` (recommended):

```env
AETHER_ENGINE=mock
AETHER_CHAOS=false
API_HOST=127.0.0.1
API_PORT=8000
```

`config.yaml` must use the **nested** schema (`model`, `strategies`, `gatekeeper`, `server`). Flat top-level keys are ignored by the loader. Set `server.engine: mock` as a second lock if desired.

### Tests

```bash
export AETHER_ENGINE=mock
export AETHER_CHAOS=false
python -m pytest tests/ -v
```

### Run control plane + demo agent

Terminal 1:

```bash
export AETHER_ENGINE=mock
export AETHER_CHAOS=false
uvicorn src.server:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
python scripts/safe_agent.py
```

Inspect:

```bash
tail -f logs/aetherforge.log
tail -f logs/hardware_safety.log
```

---

## Architecture (current)

```
Agent (safe_agent.py / your stack)
        │
        ▼
FastAPI control plane
  ├─ Economic Gatekeeper (swap ROI)
  ├─ Hardware monitor + async watchdog
  ├─ Thermal / VRAM lock → 503
  ├─ Context ceiling → 413
  ├─ Rotating ops + safety logs
  └─ Engine factory (lazy)
        ├─ mock          ← default / CI / no-GPU
        ├─ llama         ← experimental real path
        └─ ktransformers ← stub
```

---

## API surface (agent-facing)

| Method | Path | Role |
|--------|------|------|
| GET | `/system/tools` | OpenAI-compatible tool schema for strategy requests |
| GET | `/system/metrics` | Strategy, thermal lock, VRAM pressure, vitals, TPS baselines |
| GET | `/system/cache` | Cache / strategy / engine availability snapshot |
| POST | `/system/strategy` | Request mode change; Gatekeeper may reject |
| POST | `/generate` | Generate with optional strategy; respects locks and limits |

**Agent contract:** treat **503** as temporary hardware lock (backoff / wait on metrics), **413** as “shrink context,” and Gatekeeper `rejected` as “keep current mode.”

---

## Real engine (experimental — not default)

Only if you accept unproven Fast-Swap risk on your hardware:

```bash
export AETHER_ENGINE=llama
# CUDA-enabled llama-cpp-python required
# GGUF path must exist or boot fails hard (no silent Mock fallback when forced to llama)
```

Requirements for a future Wedge B experiment (not this sprint): written protocol, small model / low layer counts, measured VRAM, explicit abort rules. Until then, leave the muscle dark.

---

## Project layout

```
src/
  server.py              # Control plane, watchdog, routes
  config.py              # Nested YAML + env settings
  logger.py              # Console + rotating ops/safety logs
  hardware_monitor.py    # NVML or simulated vitals (+ optional chaos)
  engines/
    __init__.py          # Lazy create_engine
    base.py              # BaseAetherEngine contract
    mock_engine.py       # Headless muscle
    llama_engine.py      # Experimental real muscle
  ...
scripts/safe_agent.py    # Reference external agent
tests/test_hypervisor.py # Mock control-plane tests
logs/                    # Created at runtime (gitignored)
```

---

## Roadmap

**This sprint (Wedge A — continue)**  
- Keep Mock as the default story and CI path  
- Harden docs, agent examples, and 503/413 client patterns  
- Improve observability and Gatekeeper messaging honesty  
- No unsupervised real Fast-Swap on the 4060  

**Later (Wedge B — gated research)**  
- Measured Fast-Swap + KV survival protocol on consumer GPUs  
- Stronger VRAM release / process isolation if bindings leak  
- Only then reconsider “memory hypervisor” as a primary claim  

**Research**  
- True dynamic expert placement (separate branches; not production path)

See `ROADMAP.md` and `PROJECTGUIDELINES.md` for process detail; prefer this README when they conflict with verified status.

---

## Contributing

- Keep `main` runnable under **Mock** without a GPU.  
- Prefer short-lived feature branches and conventional commits.  
- Real-engine experiments stay opt-in and documented as unverified until measured.  
- Do not commit `logs/` or local GGUF weights.

---

## License

MIT
```

---
