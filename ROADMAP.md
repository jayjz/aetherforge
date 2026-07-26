# AetherForge Roadmap

**Last updated: 2026-07-26**

AetherForge provides a hardware-aware **safety and control plane** for local AI agents, with a gated research path toward measured strategy switching (Fast-Swap / KV survival) on consumer GPUs.

**Default product story (Wedge A):** Mock-runnable control plane — Gatekeeper, thermal/VRAM circuit breakers, context limits, tool discovery, audit logs, reference agent client.  
**Research story (Wedge B):** Measured real-engine Fast-Swap and related kernel work — not the default claim until proven.

---

## Current Status (main — Wedge A)

**Verified on main under Mock:**
- [x] FastAPI control plane + Pydantic schemas
- [x] Economic Gatekeeper (accept / reject strategy swaps)
- [x] Thermal / VRAM watchdog → emergency lock → **503**
- [x] Context ceiling → **413**
- [x] OpenAI-compatible tool schema (`GET /system/tools`)
- [x] Rotating ops + safety logging
- [x] Hypervisor pytest suite (discovery, Gatekeeper matrix, thermal lock)
- [x] Reference client `scripts/safe_agent.py`
- [x] Lazy engine factory (Mock without CUDA imports)
- [x] README aligned to verified reality

**Explicitly experimental / not verified:**
- [ ] Real `LlamaEngine` Fast-Swap with reliable VRAM teardown on 8 GB
- [ ] KV-cache survival across real swaps under load
- [ ] Production ktransformers / HOBBIT dynamic experts

---

## Phase A — Safety Control Plane (active sprint)

**Goal:** Make the agent-facing safety layer boringly reliable and honestly documented.

- [x] Mock engine + headless development path
- [x] Circuit breakers and Gatekeeper under test
- [x] Structured logging
- [x] README honesty pass
- [ ] PROJECTGUIDELINES + ROADMAP + CHANGELOG fully aligned (this update)
- [ ] Stronger agent client patterns (retry/backoff examples on 503)
- [ ] CI green on Mock-only install path
- [ ] Gatekeeper log copy accuracy (e.g. same-mode reject messaging)
- [ ] Optional: expand tests (concurrency, strategy+generate under lock)

**Exit:** A stranger can Mock-boot, test, run `safe_agent.py`, and understand 503/413 without reading source. No doc claims verified Fast-Swap on 8 GB.

---

## Phase B — Measured Real Engine (gated)

**Goal:** Prove or disprove Fast-Swap + KV survival on disclosed consumer hardware under a written protocol.

**Entry criteria:** Phase A exit met; experiment doc exists (model, layers, abort rules, metrics).

- [ ] Tiny/small GGUF protocol (not DeepSeek-first)
- [ ] Measured VRAM before/after swap; documented teardown behavior
- [ ] Recovery paths (fallback strategy, hard fail semantics)
- [ ] Process isolation if bindings leak memory
- [ ] Benchmarks with full hardware disclosure

**Exit:** Either (1) evidence-backed real-engine guide with known limits, or (2) explicit decision to keep real Fast-Swap research-only.

---

## Phase C — Agent Ecosystem (after B evidence or parallel on Mock)

**Goal:** Deeper agent integrations on the **verified** safety contract.

- [ ] Production-shaped examples (LangGraph / CrewAI / raw tool-calling) using 503/413 semantics
- [ ] Richer metrics and operator runbooks
- [ ] Packaging polish (pip/Docker) with Mock default

**Exit:** External agent stacks can adopt the control plane without custom folklore.

---

## Later / Research

- True dynamic expert placement (HOBBIT-inspired)
- Deeper ktransformers integration (if maintained and justified)
- KV compression, disk-tier caching
- Broader hardware profiles (with disclosure discipline)

These items do **not** redefine the default product claim until they pass the same honesty bar as Phase B.

---

## Versioning & Release Policy

- `0.x`: Iterative; breaking changes documented in CHANGELOG
- Tag when behavior **and** docs agree
- `main` always Mock-runnable
- Prefer README + PROJECTGUIDELINES when older roadmap text conflicts; then fix the roadmap

*Focus: ship a trustworthy safety control plane first; earn the right to claim memory-hypervisor behavior with measurements.*
