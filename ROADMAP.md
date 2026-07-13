# AetherForge Roadmap

## Vision
AetherForge aims to build an intelligent, predictive memory hypervisor for local LLM inference. It treats VRAM, system RAM, and storage as a unified smart hierarchy, enabling efficient use of larger MoE (and eventually dense) models on consumer hardware like RTX 4060 (8GB VRAM) + 32GB RAM through dynamic, workload-aware decisions on expert loading, mixed precision, prefetching, and caching.

The project starts narrow (MoE expert management MVP) and evolves into deeper agent integration and full hypervisor capabilities, directly supporting practical agent systems like OpenClaw.

## Guiding Principles
- **Narrow scope first**: Ship measurable MVP before expanding.
- **Build on existing**: Leverage llama.cpp MoE offloading (--n-cpu-moe, -ot overrides), HOBBIT techniques, and learnings from Colibri fork.
- **Measure everything**: Benchmarks on real hardware (user's 4060 + 32GB) are core.
- **Agent-centric**: Design with hooks for agent frameworks from day one.
- **Open & documented**: Clear research notes, decision logs, and honest limitations.

## High-Level Phases

### Phase 0: Setup & Research (In Progress - July 2026)
- [x] Repo creation and initial README with MVP definition.
- [x] Research HOBBIT paper, llama.cpp MoE features, Colibri fork.
- [ ] Detailed analysis of llama.cpp MoE code and community forks (ik_llama.cpp, cache discussions).
- [ ] Set up local dev environment (Python + llama-cpp-python or direct llama.cpp build).
- Deliverable: This roadmap + PROJECTGUIDELINES.md

### Phase 1: MVP Design & Python Prototype (Target: 2-4 weeks)
- Design simple predictive logic (layer similarity prefetch + gating-based importance).
- Implement Python orchestration layer that wraps llama.cpp (via bindings or subprocess) to add:
  - Basic importance scoring.
  - Predictive prefetch decisions.
  - Dynamic mixed-precision suggestions or overrides where possible.
  - Simple cache management on top of existing --n-cpu-moe / -ot.
- Expose clean API/hooks for agents (query cache state, request fidelity level).
- Initial benchmarking harness on target hardware.
- Deliverables: Working prototype script/notebook, basic benchmarks vs baseline, updated README.

### Phase 2: Benchmarking, Iteration & Validation (Target: 1-2 weeks after Phase 1)
- Rigorous testing on Mixtral-8x7B, Phi-MoE, Qwen MoE variants, etc.
- Compare against vanilla llama.cpp, basic --n-cpu-moe, and HOBBIT-inspired baselines (where replicable).
- Tune thresholds, policies (inspired by HOBBIT multidimensional cache).
- Measure: VRAM usage, tokens/sec, prefill/decode latency, accuracy/quality impact.
- Iterate on prototype based on data.
- Deliverables: Benchmark report (in repo), refined prototype.

### Phase 3: Deeper Integration & C++ Extensions (Target: 4-8 weeks)
- Explore contributing to or forking llama.cpp for native predictive features (or build custom extensions).
- Implement more advanced HOBBIT-like elements: token-level dynamic loading, adaptive prefetching with stacking, mixed-precision expert handling if not native.
- Enhance KV cache awareness if feasible.
- Improve agent integration (deeper hooks, shared state).
- Deliverables: C++ components or patches, improved performance, integration examples with OpenClaw or test agents.

### Phase 4: Agent Ecosystem Integration & Polish (Ongoing)
- Full hooks and examples for OpenClaw, Hermes, etc.
- Workload-aware adaptation (learn from agent task patterns).
- Documentation, examples, demos (tie to cinematic portfolio style).
- Potential packaging or easy install.
- Deliverables: Integration demos, comprehensive docs, v0.1 release.

### Phase 5: Future / v2+ (Longer term)
- LLM-driven meta-predictors or small auxiliary models for importance/prefetch.
- Support for dense models and full unified memory hypervisor (KV, activations, weights).
- Cross-model / multi-agent memory sharing.
- Hardware auto-detection and self-tuning.
- Community contributions and benchmarking.

## Milestones & Success Criteria
- MVP: Working Python layer with measurable gains on consumer hardware + agent hook demo.
- v0.1: Solid benchmarks, docs, and at least one real agent integration showing practical benefit.
- Long-term: Recognized as useful infrastructure for local agent builders on limited hardware.

## Risks & Mitigations
- Scope creep: Strict MVP definition and weekly scope reviews.
- Performance overhead: Profile early and often; start with heuristics before ML predictors.
- llama.cpp internals complexity: Start with Python wrapper; move to C++ only for proven wins.
- Hardware variability: Benchmark on target (4060 + 32GB) primarily; note generalizability.

## Next Actions
1. Finalize and commit ROADMAP.md + PROJECTGUIDELINES.md.
2. Deep dive into specific llama.cpp MoE source files and HOBBIT paper details.
3. Set up dev environment and first prototype skeleton.
4. Weekly progress updates in issues or a dedicated tracking file.

*This roadmap is living — update as we learn.*