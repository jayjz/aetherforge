# 🗺️ AetherForge Roadmap

**Last updated: 2026-07-17**

AetherForge is an agent-aware memory hypervisor for local MoE inference on consumer hardware.  
Current production path: controlled Fast-Swap (model reload with different layer counts) + KV-cache serialization + Economic Gatekeeper + OpenAI-compatible tool schema.

We ship the reliable path first. True in-memory expert routing remains a research track.

---

## Current Status (v0.3.x)

**Working today**
- [x] Control plane (FastAPI + Pydantic)
- [x] Strategy modes (`high_fidelity` / `balanced` / `aggressive_quant`)
- [x] KV-cache survival across Fast-Swaps
- [x] Economic Gatekeeper (rejects unprofitable swaps)
- [x] Dynamic OpenAI tool schema generation from Pydantic
- [x] Minimal ReAct-style agent discovery test

**Still hard-coded / fragile**
- Model path, strategy → layer mapping, Gatekeeper numbers, `n_ctx`
- No configuration system
- Limited observability
- Documentation lag

---

## Phase A — Production Hardening (Next, v0.4.0)

**Goal**: Make the current system something a stranger can clone, configure, and depend on.

- [ ] Configuration system (YAML/ENV) for model path, strategies, Gatekeeper thresholds, `n_ctx`, ports
- [ ] Single source of truth for tool schema (already started) + client library consistency
- [ ] Structured logging + basic metrics endpoint
- [ ] Proper error handling and graceful degradation
- [ ] Automated tests (unit + the existing empirical scripts) + minimal CI
- [ ] Updated README with honest setup instructions and known limitations
- [ ] Semantic versioning + CHANGELOG

**Exit criteria**: A new user can install, point at their own GGUF, change strategies via the tool, and get reproducible behavior without editing source.

---

## Phase B — Real Agent Integration (v0.5.0)

**Goal**: Prove value inside actual orchestrators.

- [ ] Clean LangGraph / CrewAI / n8n integration examples
- [ ] Live measured t/s fed back into Gatekeeper (replace static numbers)
- [ ] Optional MCP server surface
- [ ] Telemetry usable by agents (current strategy, last swap cost, VRAM pressure)

**Exit criteria**: At least one non-trivial agent loop demonstrably benefits from autonomous strategy changes with long context.

---

## Phase C — Research Track (parallel, not blocking)

**Goal**: Explore true dynamic expert movement and advanced caching.

- [ ] Investigate / prototype hot-expert caching or runtime tensor movement (requires careful evaluation of llama.cpp / ggml-backend capabilities)
- [ ] Mixed-precision expert handling (HOBBIT-inspired)
- [ ] Asynchronous pre-fetch experiments
- [ ] Cross-platform (MLX / Metal) exploration

These live on `research/*` branches. Successful results are extracted cleanly into `main` only after they are stable and configurable.

---

## Later

- Multi-model orchestration
- KV compression / relevance-based eviction
- Packaging (pip, Docker)
- Broader hardware profiles

---

## Versioning Policy

- `0.x` = pre-1.0, breaking changes allowed with notes
- Tag every meaningful release on `main`
- Keep `main` always runnable
