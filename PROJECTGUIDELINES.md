# PROJECTGUIDELINES.md — AetherForge

**Living guidelines for architecture, decisions, and engineering standards.**  
*Last updated: 2026-07-19*

## 1. Vision & Strategy

**Vision**: AetherForge is the premier **agent-aware hypervisor** for elastic local MoE inference. It enables autonomous agents to dynamically manage VRAM/experts on consumer hardware (8–24 GB VRAM), minimizing prefill penalties and maximizing throughput for long-context, multi-step workflows.

**Core Differentiator**: Combines reliable control-plane orchestration (Economic Gatekeeper, strategy switching) with high-performance heterogeneous kernels (via ktransformers integration) and cutting-edge research (HOBBIT-inspired mixed-precision dynamic experts).

**Integration Strategy**: 
- Production: Fast-Swap + KV survival + Gatekeeper.
- Accelerated: ktransformers CPU/GPU expert scheduling + FP8/AMX optimizations.
- Research: Full HOBBIT-style token-level dynamic loading, prefetch, and multidimensional caching.

**Primary Success Metrics (Product-Oriented)**:
- New user installs and runs an agent-driven strategy/expert change in <10 minutes.
- ≥2x effective agent throughput (prefill + decode) vs. static baselines on 8-16GB GPUs.
- Seamless integration with ≥3 agent frameworks (LangGraph, CrewAI, SGLang).
- 500+ GitHub stars within 6 months through usability and demonstrated value.
- Zero critical bugs in production paths; measurable accuracy preservation in dynamic modes.

## 2. Architecture Principles

1. **Reliability & Agent-First** — Production paths prioritize stability and discoverability (OpenAI tool schemas, telemetry). Research only lands on `main` when stable/configurable.
2. **Separation of Concerns** — Python control plane (decisions, Gatekeeper) vs. high-perf kernels (ktransformers/llama.cpp C++/CUDA).
3. **Config-Driven & Observable** — All knobs in YAML/ENV. Real EMA telemetry over static tables.
4. **Pragmatic Hybrid** — Leverage ktransformers for expert offload/scheduling; extend with HOBBIT techniques.
5. **Usability & Reproducibility** — Docker-first, auto-detection, honest benchmarks, minimal source edits.

## 3. Branching, Releases & Collaboration

- `main` always runnable and production-grade.
- Short-lived branches: `feat/`, `fix/`, `chore/`, `docs/`, `research/`.
- All changes via PR (self-review + rationale even solo). Conventional commits.
- Semantic versioning + `CHANGELOG.md`. Tag meaningful releases.
- Research on `research/*`; extract cleanly after validation.
- **Contributors**: See `CONTRIBUTING.md`. Welcome issues/PRs. Maintainers triage promptly.

## 4. Coding & Quality Standards

- Python 3.10+ (control plane), type hints, Pydantic v2.
- C++/CUDA for kernels (align with ktransformers style).
- Comprehensive tests: unit + integration (empirical scripts) + CI (GitHub Actions).
- No magic numbers/hard-coded paths. Full configuration.
- Security: Input validation, resource limits, no untrusted model execution.
- Documentation: Update README, ROADMAP, this file with every phase change.

## 5. Decision Log (Append-Only)

**2026-07-19** — Adopted unified strategy: AetherForge as agent hypervisor leveraging ktransformers kernels + HOBBIT research. Prioritize packaging, integrations, and benchmarks for community traction.  
**2026-07-17** — (Prior entries retained)...

## 6. Research References

- HOBBIT (arXiv:2411.01433): Mixed-precision dynamic loading, prefetch, caching.
- ktransformers: Heterogeneous MoE scheduling, FP8, SGLang integration.
- llama.cpp ecosystem & community experiments (expert cache, hybrid offload).

## 7. Testing, Benchmarking & Release Criteria

- Hardware disclosure mandatory in PRs/benchmarks.
- Baselines: vanilla llama.cpp, static offload, prior versions.
- Metrics: t/s (prefill/decode), swap/expert load latency, context fidelity, Gatekeeper decisions, accuracy (where applicable).
- Before "production": Docker support, end-to-end agent examples, CI passing, updated docs.

## 8. Packaging, Community & Portfolio Standards

- Deliverables: Docker, pip, HF demos.
- Community: Responsive issues, Discord/Reddit/X engagement, clear `CONTRIBUTING.md`.
- Professionalism: Clean history, professional READMEs, reproducible benchmarks.

## 9. Open Risks & Mitigations

- Kernel maintenance (mitigate via upstream contributions).
- Hardware fragmentation (auto-config + profiles).
- Research instability (strict gating to main).

*This is a living document. Update when reality changes.*
