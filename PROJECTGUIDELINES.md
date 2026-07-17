# PROJECTGUIDELINES.md — AetherForge

**Living guidelines for architecture, decisions, and engineering standards.**  
*Last updated: 2026-07-17*

## 1. Vision & Current Reality

**Vision**: Agent-aware memory hypervisor that lets local agents treat large MoE models as elastic resources on consumer hardware (8–16 GB VRAM + system RAM).

**Current production path** (what ships on `main`):
- Controlled Fast-Swap via `n_gpu_layers` change
- KV-cache serialization (`save_state` / `load_state`) so context survives
- Economic Gatekeeper that rejects swaps whose latency cost exceeds expected benefit
- OpenAI-compatible tool schema generated from Pydantic
- FastAPI control plane

True runtime expert-level tensor movement remains research until proven stable.

**Primary success metrics (product-oriented)**:
- A new user can install and run a strategy change with long context in < 15 minutes of setup.
- Measurable reduction in prefill penalty for multi-step agent workloads vs static baselines.
- Tool schema works with at least two major agent frameworks without custom glue.
- No silent context loss or segfaults under normal use.

## 2. Architecture Principles

1. **Reliability over cleverness** — Prefer a well-tested teardown + KV path over fragile ctypes experiments.
2. **Control plane vs muscle** — Keep decision logic (Python) separate from execution (llama.cpp / future backends).
3. **Agent-first** — Every capability should be discoverable and callable by an autonomous agent.
4. **Config over code** — Hard-coded paths, layer counts, and thresholds are technical debt.
5. **Honest observability** — Expose real measured costs, not just static tables.

## 3. Branching & Release Workflow

- `main` is always runnable.
- Work on short-lived branches: `feat/`, `fix/`, `chore/`, `docs/`, `research/`.
- Open a PR even when solo (self-review + permanent rationale).
- Delete branches after merge.
- Tag releases with semantic versioning (`v0.4.0` …).
- Long-running speculative work stays on `research/*` and is periodically rebased or archived.

## 4. Coding & Quality Standards

- Python 3.10+, type hints, Pydantic for all external interfaces.
- No new hard-coded model paths or magic numbers. Use configuration.
- Tests: unit tests for pure logic + the existing empirical scripts (KV, strategy, tool calling).
- Conventional commits.
- Update ROADMAP.md and this file in the same PR when phase status changes.

## 5. Decision Log (append-only)

**2026-07-13** — Initial decisions: Python-first control plane on top of llama.cpp, focus on MoE, agent hooks early, RTX 4060 as primary target.

**2026-07-17** — Adopted teardown + KV serialization as the production Fast-Swap mechanism. True dynamic expert movement moved to research track. Schema generation made dynamic from Pydantic. Economic Gatekeeper made the authority for swap approval.

*(Add new dated entries here.)*

## 6. Research References (kept for context)

- HOBBIT (arXiv:2411.01433) — mixed-precision expert offloading, importance scoring, prefetching.
- llama.cpp `--n-cpu-moe` / `--override-tensor` — current practical hybrid MoE baseline.
- Community hot-expert cache experiments and disk-paging PoCs.

These inform the research track. They do not define the current production path.

## 7. Testing & Benchmarking

- Always record hardware (GPU, VRAM, RAM, CUDA version).
- Baselines: vanilla llama.cpp, static `--n-cpu-moe`, previous AetherForge version.
- Metrics: swap latency, post-swap t/s, context survival, Gatekeeper accept/reject rate.
- Prefer scripts that can be re-run by others.

## 8. Packaging & Usability Goals

Before calling anything “production”:
- Configuration file or environment variables for all important knobs.
- Clear README with known limitations.
- Minimal reproducible example for agent integration.
- No requirement to edit source code for normal use.

## 9. Open Risks

- Full model reload (even with KV restore) is still relatively expensive.
- Reliance on llama-cpp-python behavior across versions.
- Local models vary widely in tool-calling reliability.
- True dynamic tensor movement may require a custom llama.cpp patch or fork.

---

*This is a living document. Update it when reality changes.*
