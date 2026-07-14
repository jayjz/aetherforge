# 🗺️ AetherForge Roadmap

AetherForge is evolving from a static inference wrapper into an **Agent-Aware Dynamic Hypervisor**. Our MVP strategy focuses on **Phase-Based Layer Orchestration**—allowing autonomous agents to hot-swap MoE layers between VRAM and RAM mid-session based on task priority.

---

## 🟢 Phase 1: The Control Plane (Completed)
Establishing the pure-Python intelligence to map and predict memory usage without touching CUDA.

- [x] **Introspection Layer:** Safe parsing of GGUF headers to map MoE topology (`model_analyzer.py`).
- [x] **The Brain:** Predictive VRAM mathematics tracking layer/expert memory footprints (`cache_manager.py`).
- [x] **API Gateway:** FastAPI server listening for agent strategy commands.
- [x] **Environment Resiliency:** CPU/Simulation fallback for cross-platform development.

## 🟡 Phase 2: The Tensor Bridge (Current Focus)
Connecting the API's strategy to the C++ hardware engine using Dynamic Layer Orchestration.

- [x] Define abstract `ITensorBridge` and immutable `MemoryInstruction` payloads.
- [ ] **Dynamic Reallocation Hack:** Map `ctypes` bindings to forcefully alter `llama_context` memory blocks.
- [ ] **Phase-Based Layer Swapping:** Implement Option B (shifting blocks of MoE layers in/out of VRAM based on Agent Mode: `HIGH_FIDELITY` vs `AGGRESSIVE_QUANT`).
- [ ] **Latency Mitigation:** Implement asynchronous pre-fetching (loading Layer block B while the GPU computes Layer block A).

## ⚪ Phase 3: Agent Integration
Wiring the hypervisor directly into standard agent workflows.

- [ ] Build the `AetherForge Tool` for standard frameworks (LangChain / AutoGen / n8n).
- [ ] Implement automatic "Tool-to-Strategy" mapping (e.g., Code Execution tool automatically triggers `HIGH_FIDELITY` memory state).
- [ ] Expose live hardware telemetry (VRAM/Temp) to the Agent's system prompt context.

## ⚪ Phase 4: Beyond the MVP
- [ ] Support for MLX Engine (Apple Silicon / Mac Unified Memory).
- [ ] Custom KV-Cache compression based on context relevance.
- [ ] Multi-model orchestration (swapping out small routing models for large reasoning models in a single VRAM pool).