# AetherForge Client SDK

This is the auto-generated, strictly-typed Python client for the AetherForge Hardware Hypervisor API. 

### Installation

\\\ash
pip install "git+https://github.com/jayjz/aetherforge.git#subdirectory=sdk/python"
\\\

### Usage

\\\python
from aetherforge_hypervisor_api_client import Client
from aetherforge_hypervisor_api_client.api.default import get_metrics_system_metrics_get

client = Client(base_url="http://127.0.0.1:8000")
metrics = get_metrics_system_metrics_get.sync_detailed(client=client)
print(metrics.content)
\\\
"@

# 7. Write the Root README.md
Write-Host "Writing root README.md..."
Set-Content -Path README.md -Encoding UTF8 -Value @"
# 🛡️ AetherForge

**The Backpressure & Safety Gateway for Local Multi-Agent Swarms.**

AetherForge sits between autonomous agent frameworks (LangGraph, CrewAI, AutoGen) and local inference runtimes (llama.cpp, vLLM, Ollama). It prevents runaway loops from triggering CUDA Out-of-Memory (OOM) crashes or thermal throttling by enforcing Agent-Infrastructure Negotiation.

---

### ⚡ 3-Minute Quickstart (Headless Mock Mode)

#### 1. Launch the Control Plane & Operator Lens
\\\ash
docker compose up --build
\\\
* **API Gateway:** http://localhost:8000
* **Real-time Telemetry Dashboard:** http://localhost:8501

#### 2. Run the LangGraph Fallback Showcase
Open a new terminal and run the showcase. You will need an OPENAI_API_KEY exported in your environment.
\\\ash
# Install the AetherForge SDK from the local directory
pip install -e sdk/python

# Run the showcase
python examples/02_langgraph_recovery.py
\\\

> **What you'll see:** The agent sends a 40,000-character payload, the API intercepts it with an HTTP 413 Context Exceeded error to protect the VRAM. LangGraph's topology catches the machine-readable error, truncates the context autonomously, and completes the execution without a server crash.

---

### Architecture
> **Verified Today (Wedge A):** The safety control plane, Economic Gatekeeper, concurrent queue admission, and structured 503/413 circuit breakers are verified under AETHER_ENGINE=mock. This allows agent resilience testing without requiring dedicated CUDA hardware.
>
> **Experimental (Wedge B):** Physical Fast-Swap with reliable VRAM teardown and KV-cache survival. These components have been quarantined to src/research_archive/ and are not on the production hot-path.
