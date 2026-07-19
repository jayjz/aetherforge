# Changelog

All notable changes to AetherForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] - 2026-07-19

### Added
- **Dynamic YAML Configuration**: Implemented a custom YAML flattener utilizing Pydantic in `src/config.py` to safely parse `config.yaml` with a robust fallback architecture.
- **Hardware Profile Presets**: Created `config.yaml.example` specifically calibrated for consumer-grade rigs (RTX 4060 8GB VRAM + 32GB RAM).
- **Hardened Containerization**: Introduced a modernized `Dockerfile` enforcing `GGML_CUDA=on` to guarantee native GPU acceleration, along with a production-ready `docker-compose.yml`.
- **Harness Validation**: Added `scripts/test_engine_abstraction.py` to automatically assert and enforce telemetry dictionary shapes and types across all base engine contracts.

### Changed
- **Engine Factory Architecture**: Decoupled the FastAPI control plane from specific backend runtimes by wrapping backends in an abstraction layer (`BaseAetherEngine` interface via `create_engine` factory pattern).
- **Directory Topology**: Structured backend isolation by migrating inference layers into isolated modules under `src/engines/`.

### Fixed
- **Contract Enforcement**: Refactored `MockAetherEngine` to explicitly inherit from `BaseAetherEngine`, resolving integration gaps with the automated contract test suite.

---

## [0.4.2] - 2026-07-17

### Added
- **Silicon Telemetry**: Integrated `HardwareMonitor` backed by `pynvml` to pull live physical GPU temperature and allocation boundaries.
- **Circuit Breakers**: Added 503 Service Unavailable route-level thermal blocks protecting the GPU against runaway loads.
- **Proactive Preflights**: Introduced a token count pre-check on the `/generate` endpoint, returning a 413 Payload Too Large error before ingestion when thresholds are breached.
- **State Mutual Exclusion**: Added an asynchronous execution lock (`asyncio.Lock`) inside `server.py` to guarantee thread-safe VRAM strategy transitions.

### Changed
- **Telemetry Processing**: Connected learned wall-clock serialization costs and tokens-per-second (TPS) values to the `EconomicGatekeeper` via Exponential Moving Average (EMA) tuning.

---

## [0.4.0] - 2026-07-06

### Added
- **Control Plane Core**: Established core FastAPI infrastructure, custom Pydantic schemas, and initial `EconomicGatekeeper` evaluation logic.
- **Cache Management**: Deployed baseline `AetherCacheManager` to track tensor locations across physical RAM and VRAM.