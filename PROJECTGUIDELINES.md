# PROJECTGUIDELINES.md - AetherForge

**Extensive guidelines, research synthesis, architecture, and development standards for AetherForge.**

*Last updated: July 13, 2026*

## 1. Project Vision & Core Goals
AetherForge is a predictive, intelligent memory hypervisor for local LLMs. It dynamically manages the memory hierarchy (VRAM on limited GPUs like RTX 4060 8GB, system RAM 32GB, and fast storage) for MoE models primarily, with future extension to dense models and KV/activation management.

**Primary Goal (MVP)**: Build a practical prototype that improves effective model capacity and/or speed on consumer hardware through smarter expert loading/prefetching/caching compared to static baselines, while providing clean integration points for agent systems.

**Longer-term Vision**: Become foundational infrastructure that lets agent builders (like OpenClaw) treat large MoE models as first-class citizens on modest hardware by making memory decisions automatic, predictive, and workload-aware.

**Success Metrics**:
- Measurable improvement in VRAM efficiency or tokens/sec on target hardware vs. vanilla llama.cpp / basic offloading.
- Quality impact <1-2% on standard benchmarks.
- Usable hooks for agent frameworks.
- Clear, reproducible documentation and benchmarks.

## 2. Research Synthesis & Key References

### HOBBIT Paper (Primary Inspiration)
**Title**: HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference (arXiv:2411.01433)

**Core Ideas to Adopt/Adapt**:
- **Importance Scoring**: Use gating input magnitude ||G(x)_e_i|| as proxy for expert importance (highly correlated with actual contribution). Compute unimportance score cumulatively. Thresholds (e.g., T1=0.6, T2=0.9) to decide high-prec / low-prec / skip.
- **Mixed Precision**: Load less important experts in int4/int2 instead of float16 to reduce I/O latency (up to 4x faster loading) with minimal accuracy loss if limited percentage.
- **Layer-level Adaptive Prefetching**: Exploit high cosine similarity of gating inputs between consecutive layers (~0.9+). Predict next-layer experts with high accuracy (~96% top-1 for next layer). Use stacking for efficient multi-layer prediction. Mitigate wrong predictions with low-precision loads.
- **Multidimensional Caching**: Combine LRU, LFU, LHU (high-prec frequently used), FLD (farthest layer distance). Weighted priority to minimize mixed miss penalties.
- **Implementation**: Built on llama.cpp (~8k LOC C++/C additions). GPU-centric and CPU-GPU modes.

**Results (Reference)**: Significant speedups (up to ~10x decoding on edge devices) with <1% accuracy drop on Mixtral-8x7B and Phi-MoE.

**Link**: https://arxiv.org/abs/2411.01433 (or HTML version for reading).

### llama.cpp MoE Support (Current Foundation)
llama.cpp provides practical MoE offloading:
- `--cpu-moe` or `--n-cpu-moe N`: Keeps expert weights from (parts of) layers in CPU/RAM.
- `-ot "pattern=CPU"` or `--override-tensor`: Fine-grained tensor placement (e.g., specific layers' exps to CPU).
- `-ngl N`: GPU layers.
- Community work: Discussions on two-tier GPU+RAM expert caches, eviction policies, --moe-cache flags in forks, ik_llama.cpp hybrid enhancements.
- Recent: SYCL fused top-k MoE, etc.

**Strategy for AetherForge**: Start by intelligently deciding *which* experts/layers to offload via these flags or by wrapping the engine, adding predictive logic on top. Later contribute or extend for native dynamic/mixed-prec support.

**Key Files to Study** (in llama.cpp):
- MoE-related in src/ (llama.cpp, ggml*.c for tensor handling).
- Conversion scripts for MoE models.
- CLI/server code for flag parsing.

### Colibri Fork Learnings (user's jayjz/colibri)
- Extreme disk-streaming for massive MoE (GLM-5.2 744B) on ~25GB RAM.
- Features like router-lookahead prefetch, auto expert cache sizing based on available RAM, KV persistence, OpenAI-compatible API.
- Pure C core, Windows/Metal/CUDA support in fork.
- Emphasizes quality preservation and resource policies.

**Relevance**: Techniques for auto cache adaptation and prefetch ideas can inspire AetherForge heuristics. AetherForge focuses more on predictive intelligence and agent integration rather than extreme scale.

### Other Relevant Areas
- Speculative decoding & MTP (in Colibri fork).
- KV cache management and compression.
- General LLM inference optimization (quantization, Flash Attention, etc.).
- Agent frameworks: How OpenClaw/Hermes/etc. currently call inference and manage context.

## 3. Architecture Overview (Target for MVP and Evolution)

**MVP Architecture (Python-First)**:
- **Core**: Python layer that orchestrates llama.cpp (via llama-cpp-python bindings or CLI + parsing, or direct integration later).
- **Components**:
  - **Importance Scorer**: Heuristic (gating magnitude simulation or post-hoc analysis) + future small predictor model.
  - **Predictor/Prefetcher**: Layer similarity-based or simple history-based prediction of next experts.
  - **Decision Engine**: Decides offload strategy, precision hints, cache priorities. Outputs overrides or configures the backend.
  - **Cache Manager**: Tracks state; simple LRU/LFU + importance weighting on top of llama.cpp mechanisms.
  - **Agent Interface**: API to query state, set fidelity mode, receive callbacks.
- **Data Flow**: Agent request -> Decision Engine (with context/task info) -> Configure llama.cpp run (flags/overrides) -> Execute -> Post-process/feedback for learning.

**Future Layers**:
- C++ extensions or patches to llama.cpp for lower-overhead dynamic loading/mixed-prec.
- Small auxiliary ML model for better prediction (trained on traces from target hardware).
- Deeper KV cache integration.
- Multi-agent shared memory.

**Key Constraints**: Must not add significant latency. Start heuristic, profile everything. Prioritize MoE routed experts.

## 4. Decision Log (Living Document)

**Initial Decisions (July 13, 2026)**:
- **Name**: AetherForge (ties to personal branding, "forging" better systems).
- **Scope**: Strict MVP on MoE expert management with predictive elements. No full hypervisor or dense model support initially.
- **Base**: Build on llama.cpp MoE offloading features rather than from-scratch engine (Colibri is complementary for extreme cases).
- **Language Start**: Python orchestration layer for rapid prototyping and agent integration. Move performance-critical parts to C++ later.
- **Evaluation Hardware**: Primary = user's RTX 4060 + 32GB RAM setup. Secondary = note general consumer applicability.
- **Integration Priority**: Design hooks early for OpenClaw and similar agent systems.
- **Documentation**: Heavy emphasis on research synthesis, honest benchmarks, and decision rationale.

**Future Decisions to Track**: Choice of exact MoE test models, specific thresholds/policies, whether to fork llama.cpp or wrap, etc. Update this section with date and rationale for each major choice.

## 5. Coding Standards & Style
- **Python**: PEP 8, type hints where helpful, clear docstrings. Modular components (scorer, predictor, decision, cache, agent_interface).
- **C++ (later)**: Follow llama.cpp style where extending; consistent naming, performance-focused.
- **General**: Readable over clever. Comment research inspirations. Profile before optimizing.
- **Testing**: Unit tests for components; integration tests with small MoE models; benchmark scripts.
- **Versioning**: Semantic versioning once past MVP. Changelog in repo.

## 6. Development Workflow
- **Git**: Feature branches from main. PRs with description linking to issues/research.
- **Commits**: Conventional commits (feat:, fix:, docs:, refactor:). Reference issues.
- **Issues**: Use for tasks, research questions, benchmark results. Label (MVP, research, benchmark, integration).
- **Tracking**: Update ROADMAP.md and this file regularly. Consider a PROGRESS.md or use issues/projects.
- **Environment**: Python venv or conda. llama.cpp build as needed. Document setup in README or SETUP.md.
- **CI**: Basic (lint, tests) once mature; optional GitHub Actions.

## 7. Benchmarking Methodology
- **Hardware**: Record exact specs (GPU VRAM, RAM, storage speed, CPU).
- **Models**: Start with accessible MoE like Mixtral-8x7B GGUF, Qwen MoE variants, Phi-MoE. Note quantization.
- **Baselines**: Vanilla llama.cpp (full GPU where possible), --n-cpu-moe / -ot basic offload, any replicable HOBBIT-like static version.
- **Metrics**:
  - Memory: Peak VRAM, RAM usage during inference.
  - Speed: Prefill time, decode tokens/sec (average, p50/p95).
  - Quality: Perplexity or task-specific (GSM8K, etc.) if applicable; human eval or proxy for agent tasks.
  - Overhead: Added latency from decision logic.
- **Workloads**: Simple generation + simulated agent loops (multi-turn, tool-use patterns).
- **Reproducibility**: Script everything; commit benchmark configs and results.
- **Reporting**: Tables/graphs in repo (e.g., in docs/ or benchmark_results.md). Honest about limitations and variance.

## 8. Agent Integration Guidelines
- **Hooks/API**: Expose functions to get current cache state, predicted next experts, set "fidelity mode" (high-prec bias), receive post-inference feedback.
- **Context Awareness**: Pass task type, context length estimate, importance hints from agent to decision engine.
- **Examples**: Provide minimal example integrating with a simple agent or OpenClaw module.
- **Goal**: Allow agents to treat the hypervisor as an intelligent backend rather than black-box inference.

## 9. Resources & Tools
- **Core Papers**: HOBBIT arXiv, related MoE offloading papers.
- **Codebases**: llama.cpp (https://github.com/ggml-org/llama.cpp), user's Colibri fork, ik_llama.cpp if relevant, community MoE cache discussions.
- **Models**: Hugging Face GGUF MoE models.
- **Profiling**: nvidia-smi, llama.cpp built-in logging, Python profilers.
- **Docs**: Keep this file and ROADMAP.md updated. Use clear diagrams where helpful (ASCII or images later).

## 10. Contribution & Maintenance
- Open to contributions once MVP ships.
- Issues and PRs welcome with clear descriptions.
- Maintain decision log and research notes for transparency.
- License: TBD (likely MIT or Apache 2.0 like related projects).

## 11. Risks, Assumptions & Open Questions
- **Risks**: Overhead from added logic; complexity of llama.cpp internals; hardware-specific tuning.
- **Assumptions**: Heuristic importance/prediction sufficient for MVP gains; agent workloads benefit from smarter offloading.
- **Open Questions**: Exact implementation of importance scoring without full gating access; best way to apply mixed-precision dynamically; long-term path to native llama.cpp contributions.

*This is a living document. Update frequently with new research, decisions, and learnings.*

---

**End of PROJECTGUIDELINES.md**