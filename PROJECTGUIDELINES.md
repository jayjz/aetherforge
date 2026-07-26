# PROJECTGUIDELINES.md — AetherForge

**Living guidelines for architecture, decisions, and engineering standards.**  
*Last updated: 2026-07-26*

## 1. Vision & Strategy

**Current phase (Wedge A — active):**  
AetherForge is a **hardware-aware safety and control plane** for local AI agents on consumer GPUs. It sits between agents and local inference, providing strategy negotiation, Economic Gatekeeper decisions, thermal/VRAM circuit breakers, context ceilings, tool discovery, and durable audit logs. The default, documented, and CI path is **Mock-capable** and must run without CUDA.

**Long-term direction (Wedge B / research — gated):**  
Agent-aware strategy switching with *measured* Fast-Swap and KV-cache survival on consumer hardware, plus optional heterogeneous kernels. These capabilities are **not** default product claims until proven under a written experiment protocol.

**What we do not claim on `main` without evidence:**
- Production-ready Fast-Swap with reliable VRAM teardown on 8 GB cards
- Guaranteed KV survival across strategy changes on real GPUs
- True in-memory MoE expert movement or production ktransformers integration

**Near-term success metrics (Wedge A):**
- A stranger can clone, set `AETHER_ENGINE=mock`, run tests, and exercise the control plane in under 15 minutes without a GPU
- 503 (thermal lock) and 413 (context ceiling) behavior is tested and documented for agents
- README, this file, ROADMAP, and CHANGELOG agree on verified vs experimental status
- `main` never requires CUDA to import or boot the Mock path (lazy engine loading)

**Later metrics (only after Wedge B evidence):**
- Measured Fast-Swap cost and KV fidelity on disclosed hardware
- Agent throughput comparisons vs static baselines with full methodology
- Multi-framework examples (LangGraph, CrewAI, etc.) built on the verified safety contract

## 2. Architecture Principles

1. **Safety over ambition** — Prefer abort, 503, or fallback over OOM, silent corruption, or thermal thrash. Single-GPU user hardware is treated as mission-critical.
2. **Mock-first `main`** — The control plane must be fully developable and testable headless. CUDA and real GGUF loads are opt-in.
3. **Honesty in claims** — Documentation matches verified behavior. Experimental paths are labeled experimental.
4. **Separation of concerns** — Python control plane (API, Gatekeeper, watchdog, logging) vs engine backends (Mock / Llama / future kernels).
5. **Lazy loading** — Engine implementations are imported only when selected so Mock boots without `llama-cpp-python` or CUDA bindings.
6. **Config-driven & observable** — Nested YAML + environment overrides; rotating ops and safety logs; EMA telemetry where applicable.
7. **Agent-first contracts** — OpenAI-style tool schemas, explicit 503/413 semantics, and Gatekeeper accept/reject responses agents can handle.
8. **Research stays gated** — HOBBIT-style dynamic experts, ktransformers depth, and real Fast-Swap experiments live behind flags/branches until measured.

## 3. Branching, Releases & Collaboration

- `main` is always **Mock-runnable** without a GPU and must pass the hypervisor test suite under `AETHER_ENGINE=mock`.
- Short-lived branches: `feat/`, `fix/`, `chore/`, `docs/`, `research/`.
- All meaningful changes via PR (self-review + rationale even when solo). Conventional commits.
- Semantic versioning + `CHANGELOG.md`. Tag only when docs and behavior align.
- Real-engine and kernel experiments: `research/*` or explicit opt-in env flags; do not redefine default product claims in the same PR without evidence.
- Prefer this document and README over outdated roadmap language when they conflict; then fix the outdated file in the same sprint.

## 4. Coding & Quality Standards

- Python 3.10+ for the control plane; type hints; Pydantic v2.
- No eager CUDA or engine imports in factory/package init paths used by Mock.
- Nested `config.yaml` schema only (flat legacy keys are ignored and must not be documented as live).
- Tests: at minimum Mock integration tests for discovery routes, Gatekeeper matrix, and thermal 503 enforcement. Expand before claiming production readiness for real engines.
- No magic numbers for safety limits; bound critical thresholds in settings.
- Security posture: validate inputs, enforce resource ceilings, never execute untrusted models as a side effect of boot.
- Documentation: update README, ROADMAP, CHANGELOG, and this file when phase or verified status changes.

## 5. Decision Log (Append-Only)

**2026-07-26** — Aligned PROJECTGUIDELINES and ROADMAP with Wedge A. Formalized Mock-first `main`, lazy engine loading, and gated Wedge B criteria.  
**2026-07-24/25** — Adopted **Wedge A**: ship and harden the hardware safety control plane (Mock, Gatekeeper, 503/413, logging, `safe_agent.py`). Demoted Fast-Swap / KV survival on 8 GB hardware to experimental **Wedge B** pending a written protocol and measurements. README rewritten to match verified reality. Feature branch control-plane hardening merged to `main`.  
**2026-07-23** — Control-plane hardening under Mock: engine toggles (`AETHER_ENGINE`, `AETHER_CHAOS`), sticky thermal chaos verification, contract/temperature wiring, hypervisor pytest suite.  
**2026-07-19** — Adopted unified long-range strategy: agent hypervisor direction leveraging ktransformers kernels + HOBBIT research. Packaging and benchmarks prioritized for future traction. *(Historical intent; not current default product claim.)*  
**2026-07-17** — Thermal watchdog, circuit breakers, and related safety scaffolding introduced on the control plane.

## 6. Research References

- HOBBIT (arXiv:2411.01433): Mixed-precision dynamic loading, prefetch, caching — research track only until implemented and measured.
- ktransformers: Heterogeneous MoE scheduling — stub/adapter only on `main`; not a production dependency.
- llama.cpp / llama-cpp-python: Fast-Swap and state save/load experiments — opt-in engine path.

## 7. Testing, Benchmarking & Release Criteria

### Wedge A cut (current bar for `main`)

- [ ] `AETHER_ENGINE=mock` boots on a machine without a GPU
- [ ] `python -m pytest tests/ -v` passes
- [ ] README and this file do not claim verified Fast-Swap/KV survival on 8 GB
- [ ] 503 thermal lock and Gatekeeper reject paths remain covered by tests
- [ ] Real engine remains opt-in and labeled experimental
- [ ] Logs directory is gitignored; no secrets or GGUF weights committed

### Wedge B / real-engine claims (additional bar)

- [ ] Written experiment protocol (model size, layer counts, abort rules, hardware disclosure)
- [ ] Measured VRAM behavior across swaps; no reliance on unproven teardown myths
- [ ] Documented failure modes (OOM, blank KV, recovery path)
- [ ] Hardware disclosure in any benchmark or PR claiming speedups

### General

- Hardware disclosure mandatory in PRs that touch real engines or publish numbers.
- Baselines and methodology required before comparative claims.
- Prefer honest “unverified” over optimistic marketing.

## 8. Packaging, Community & Portfolio Standards

- Default demos and docs lead with Mock + `scripts/safe_agent.py`.
- Docker and CUDA images are optional accelerators, not the definition of “works.”
- Portfolio and public posts must match verified status (safety control plane first).
- Clean history, professional docs, reproducible steps for headless install.

## 9. Open Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Real Fast-Swap OOM / driver mess on 8 GB | Keep `llama` opt-in; require experiment protocol; prefer process isolation later |
| Docs drift from code | Phase changes update README + this file + ROADMAP + CHANGELOG together |
| Eager imports break no-GPU CI | Lazy factory; Mock-first tests |
| Agents treat 503 as fatal | Document agent contract; improve client backoff examples |
| Scope creep into unproven hypervisor claims | Wedge A sprint discipline; research branches for ambition |

*This is a living document. Update when reality changes.*
