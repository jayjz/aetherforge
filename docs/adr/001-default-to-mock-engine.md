# ADR 001: Default to Mock Engine for Boot and CI

## Status
Accepted (Wedge A)

## Context
AetherForge relies on heavy inference bindings (llama.cpp, CUDA, NVML). Booting the server on a machine without an NVIDIA GPU, or running CI in GitHub Actions, causes fatal import crashes.

## Decision
The `create_engine` factory in `src/engines/__init__.py` will default to instantiating `MockAetherEngine` unless a valid model path and CUDA environment are explicitly detected and verified. The real `LlamaEngine` is gated behind Wedge B research.

## Consequences
- **Positive:** The FastAPI control plane, Gatekeeper math, and Watchdog logic can be developed, tested, and CI-verified on any M1/M2 Mac or GitHub Actions runner.
- **Negative:** Developers must explicitly opt-in to test real hardware VRAM swapping. 