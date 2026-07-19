# 🗺️ AetherForge Roadmap

**Last updated: 2026-07-19**

AetherForge delivers agent-optimized elastic MoE inference by combining intelligent hypervisor control with high-performance heterogeneous kernels and dynamic expert research.

**Unified Stack**: AetherForge (orchestration) + ktransformers (kernels) + HOBBIT techniques (dynamic experts).

---

## Current Status (v0.4.x)

**Production (main)**:
- [x] FastAPI control plane + Pydantic schemas.
- [x] Strategy modes with Economic Gatekeeper + KV survival.
- [x] Dynamic OpenAI tool schema.
- [x] Basic ktransformers alignment hooks.

**In Progress**: Configuration hardening, initial HOBBIT prototypes.

---

## Phase 0: Foundation & Unification (v0.4.0 — Now)

**Goal**: Production-ready base with ktransformers integration.

- [ ] Full YAML/ENV config system.
- [ ] ktransformers kernel integration (expert scheduling, FP8/AMX).
- [ ] Docker + one-click install.
- [ ] Structured logging, metrics endpoint, auto hardware profiling.
- [ ] CI/CD + basic unit/integration tests.
- [ ] Updated docs + initial benchmarks.

**Exit**: Stranger can `docker compose up` and run agent strategy changes reproducibly.

---

## Phase 1: Dynamic Experts & Agent Excellence (v0.5.0)

**Goal**: HOBBIT-powered runtime + proven agent value.

- [ ] Token-level dynamic mixed-precision loading & prefetch (HOBBIT).
- [ ] Multidimensional expert cache policy.
- [ ] Live telemetry feedback to Gatekeeper.
- [ ] Production-grade LangGraph/CrewAI/SGLang examples.
- [ ] Extended benchmarks (consumer GPUs, agent workloads).

**Exit**: Demonstrable 2-5x agent gains; community PRs welcomed.

---

## Phase 2: Ecosystem & Scale (v0.6.0 — v1.0)

**Goal**: Broad adoption.

- [ ] Multi-model orchestration & relevance caching.
- [ ] Packaging (PyPI, HF Space, LM Studio/Ollama plugins).
- [ ] Cross-platform (ROCm, Metal via extensions).
- [ ] Advanced agent features (persistent memory, self-evolution hooks).
- [ ] Security audit + broader hardware profiles.

**Exit**: v1.0 with 500+ stars, production users, upstream contributions.

---

## Later / Research

- KV compression, disk-tier caching.
- Full custom llama.cpp fork (if needed).
- Fine-tuning integration (LLaMA-Factory via ktransformers).
- Cloud-hybrid fallback.

---

## Versioning & Release Policy

- `0.x`: Iterative, breaking changes documented.
- `1.0+`: Strict semantic versioning.
- `main` always runnable. Tag on meaningful milestones.
- Maintain `CHANGELOG.md` and update portfolio-friendly READMEs.

*Focus: Deliver usable value quickly while building toward industry-leading local agent infrastructure.*
