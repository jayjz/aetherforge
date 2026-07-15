# ⚙️ AetherForge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: MVP](https://img.shields.io/badge/status-MVP-yellow.svg)]()
[![Hardware: CUDA](https://img.shields.io/badge/hardware-CUDA-green.svg)]()

**Agent-Aware Dynamic Memory Hypervisor for Local MoE Inference**

AetherForge is an intelligent runtime layer that manages VRAM, System RAM, and GPU memory as a single unified hierarchy. By treating inference as a **state-dependent task** rather than a static process, AetherForge enables large Mixture-of-Experts (MoE) models to run efficiently on consumer hardware (e.g., RTX 4060 8GB + 32GB RAM).

---

## 🚀 The Core Pivot: Phase-Based Orchestration

Instead of relying on static memory pinning, AetherForge acts as an **overseer**. It intercepts the inference engine, allowing autonomous agents to signal their intent (e.g., *"I am entering a high-fidelity coding phase"*). The hypervisor then dynamically reallocates hardware resources in real-time to prevent OOM errors while maximizing tokens-per-second (t/s).

## System Architecture
 
graph TD;
    A[Autonomous Agent] -->|POST /strategy| B(AetherForge API)
    A -->|POST /generate| B
    B -->|Topology JSON| C{The Brain - Cache Manager}
    C -->|MemoryInstruction| D[Tensor Bridge]
    D -->|Layer-Level Swap| E[(llama.cpp / GGML Backend)]
    E -->|VRAM| F[RTX 4060 GPU]
    E -->|RAM| G[System Memory]

    ---

   ## ✨ Core FeaturesDynamic Layer Orchestration: Phase-based VRAM re-allocation (Fast-Swap) allows switching between high_fidelity, balanced, and aggressive_quant states in ~5.8 seconds.Predictive Cache Manager: Maps MoE topology and generates deterministic memory payloads to ensure VRAM is prioritized for high-importance experts.Agent-First API: FastAPI-based control plane with Pydantic validation and native OpenAI-compatible function schema export for seamless agent tool binding.Environment-Aware Failsafes: Graceful fallback between hardware-accelerated generation (Windows/CUDA) and CPU-only simulation (Linux).
   
   ## 📊 Performance Baseline (RTX 4060 8GB)
   
   Strategy,Layers in VRAM,Perf (Tokens/s),Target Use Case
High Fidelity,15,~20.0 t/s,Complex Reasoning / Coding
Balanced,10,~14.2 t/s,General Agent Interaction
Aggressive Quant,2,~12.1 t/s,Simple Summarization / Data Routing
   
   StrategyLayers in VRAMPerf (Tokens/s)Target Use CaseHigh Fidelity15~20.0 t/sComplex Reasoning / CodingBalanced10~14.2 t/sGeneral Agent InteractionAggressive Quant2~12.1 t/sSimple Summarization / Data Routing⚡ Getting StartedPrerequisitesPython 3.10+llama-cpp-python compiled with CUDA accelerationRTX 4060 (or similar 8GB+ GPU)Quick StartInitialize the Environment:Bashpip install -r requirements.txt
Boot the Control Plane:Bashuvicorn src.server:app --port 8000
Execute a Strategy Swap:Command the hypervisor to allocate VRAM for coding tasks:Bashcurl -X POST [http://127.0.0.1:8000/system/strategy](http://127.0.0.1:8000/system/strategy) \
     -H "Content-Type: application/json" \
     -d '{"mode": "high_fidelity"}'
## 🗺️ Roadmap
Phase 1: Control Plane & Topology Analysis.[x] 
Phase 2: Tensor Bridge & Dynamic Layer Swapping (Current).[ ] 
Phase 3: Autonomous Agent Tool Integration (LangChain / n8n).[ ] Phase 4: Advanced KV-Cache Compression & MLX/Mac Silicon Support.

🤝 ContributingAetherForge is built for local, agentic AI infrastructure. If you are experimenting with dynamic VRAM allocation on consumer hardware, open an issue or submit a pull request.
