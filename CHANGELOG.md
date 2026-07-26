# Changelog

All notable changes to AetherForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.6.0] - 2026-07-25

### Wedge A: Hardware Safety Layer and Mock Control Plane

**Strategic positioning:** AetherForge on `main` is positioned as a **hardware-aware safety and control plane** for local AI agents. Fast-Swap / KV-cache survival on real 8 GB GPUs is an **experimental research track (Wedge B)** and is not a verified production claim.

### Added

- **Rotating file logging (`src/logger.py`):** Replaced ad-hoc `print()` control-plane output with namespaced loggers and rotating files.
  - `logs/aetherforge.log` — API, Gatekeeper, and operational events
  - `logs/hardware_safety.log` — WARNING+ safety-oriented events (including thermal lock signals)
- **Reference agent client (`scripts/safe_agent.py`):** Discovers `/system/tools`, checks metrics, negotiates strategy, runs generation, and demonstrates handling of **503** (thermal lock) and **413** (context ceiling).
- **Hypervisor integration tests (`tests/test_hypervisor.py`):** Mock-forced coverage for system discovery routes, Gatekeeper accept/reject matrix, and thermal lock → 503 enforcement on `/generate` and `/system/strategy`.
- **Engine environment controls:** `AETHER_ENGINE` and `AETHER_CHAOS` for headless development and deterministic tests.
- **Hardware circuit-breaker enforcement:** Route-level **503** when `emergency_thermal_lock` is active; **413** when prompt tokens exceed the configured safety ceiling.

### Changed

- **Lazy engine factory:** Engine backends are imported only when selected so the Mock path can boot and test without CUDA or `llama-cpp-python` import side effects.
- **LlamaEngine contract repair:** Restored `generate()`, temperature propagation, strategy mapping, and best-effort teardown/recovery paths. Real Fast-Swap on 8 GB hardware remains **unverified**.
- **Config discipline:** Nested YAML schema is the supported configuration shape; flat legacy keys are not authoritative.
- **README:** Rewritten to match verified Mock/safety reality and to demote unverified Fast-Swap claims.

### Fixed

- Control plane no longer depends on eager CUDA/engine imports for Mock operation.
- Thermal spoof / permanent lock artifacts from earlier burn-in experiments removed from the default path.

### Notes

- Logging uses the standard library `logging` module with rotating file handlers (synchronous I/O). Suitable for current control-plane QPS; not a claim of fully asynchronous logging infrastructure.
- Merging control-plane hardening to `main` does **not** authorize unsupervised real Fast-Swap on consumer 8 GB GPUs.

---

## [0.5.0] - 2026-07-19

### Added

- Dynamic YAML configuration loading via Pydantic in `src/config.py`.
- `config.yaml.example` calibrated toward consumer profiles (e.g. RTX 4060-class).
- Docker / compose scaffolding aimed at CUDA builds.
- Engine abstraction harness scripts under `scripts/`.

### Changed

- Control plane decoupled from a single backend via `BaseAetherEngine` and `create_engine`.
- Inference backends organized under `src/engines/`.

### Fixed

- Mock engine aligned to the shared engine contract for harness checks.

---

## [0.4.2] - 2026-07-17

### Added

- `HardwareMonitor` with NVML when available.
- Route-level thermal **503** protection.
- Token pre-check **413** on oversized generation context.
- `asyncio.Lock` around strategy-sensitive execution paths.

### Changed

- Gatekeeper telemetry updated with EMA-style learning of measured latencies and TPS where wired.

---

## [0.4.0] - 2026-07-06

### Added

- FastAPI control plane core, Pydantic schemas, and initial Economic Gatekeeper logic.
- Baseline `AetherCacheManager` for tracking expert/tensor placement metadata.
