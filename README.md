# AetherForge

**Intelligent Predictive Memory Hypervisor for Local LLMs**

A unified runtime layer that treats VRAM, system RAM, and fast storage as a single smart memory hierarchy for efficient local LLM inference — especially MoE models — on consumer hardware.

### The Problem
Current tools (llama.cpp ncmoe static pinning, crude disk streaming in Colibri, research systems like HOBBIT) provide pieces but lack true predictive intelligence, automatic workload-aware decisions, dynamic mixed-precision, low-overhead on-demand expert loading, and clean integration with agent frameworks. This forces manual tuning and wastes potential on hardware like RTX 4060 (8GB VRAM) + 32GB RAM.

### MVP Goal (Narrow & Shippable)
Build a working prototype focused on **MoE expert management** that demonstrates measurable improvements in memory efficiency and/or speed on consumer hardware, with basic agent integration hooks.

## 🧠 Core Architecture: Dynamic Layer Orchestration

AetherForge does not compete with `llama.cpp` on static token generation speed. It acts as an **overseer**, intercepting the inference engine to dynamically manage memory based on agent intent.

```mermaid
graph TD;
    A[Autonomous Agent] -->|POST /strategy| B(AetherForge API)
    A -->|POST /generate| B
    B -->|Topology JSON| C{The Brain - Cache Manager}
    C -->|MemoryInstruction| D[Tensor Bridge]
    D -->|ctypes Layer Swap| E[(llama.cpp / GGML Backend)]
    E -->|VRAM| F[RTX 4060 GPU]
    E -->|RAM| G[System Memory]
    
**Core Features (MVP Scope)**
- Predictive prefetching for experts based on layer similarity (inspired by HOBBIT) + simple importance scoring (gating magnitude + heuristics).
- Dynamic mixed-precision loading (high prec for important experts, lower for others).
- Basic on-demand expert loading with prefetch (improving on static -ncmoe).
- Simple importance-based eviction/keep decisions across VRAM/RAM.
- Clean Python API / hooks so agent systems (e.g., OpenClaw) can observe memory state or request "high-fidelity mode" for specific steps.
- Benchmarks on RTX 4060 + 32GB RAM: memory footprint, tokens/sec, quality delta vs vanilla llama.cpp baselines (target: meaningful gains with <1-2% quality loss).

**Non-Goals for MVP (strict)**
- Full dense model support.
- LLM-driven meta-predictors (add in v2).
- Advanced KV cache compression or full unified hypervisor.
- Production polished drop-in (focus on working prototype + clear docs).

### Why This Matters
Enables stronger agentic workflows on limited hardware by making larger effective MoE models practical without killing interactivity.

### Tech Stack (MVP)
- Base: llama.cpp (or bindings)
- Language: Python orchestration layer + C++ extensions where needed
- Evaluation: Standard benchmarks + custom agent workloads

### Status
Early MVP — tracking progress in issues.

### Getting Started / Contributing
(See installation below)

Built as part of practical local AI agent infrastructure.
