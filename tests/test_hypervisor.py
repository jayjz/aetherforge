"""
AetherForge Hypervisor Integration Tests
========================================
Executes pure logic validation against the control plane routes using
the MockAetherEngine fallback profile. Ensures deterministic testing
by forcing headless mock variables.
"""

import sys
import os
import asyncio
import pytest
from fastapi.testclient import TestClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force strict testing environment BEFORE importing server logic
os.environ["AETHER_ENGINE"] = "mock"
os.environ["AETHER_CHAOS"] = "false"

from src.server import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_system_discovery_endpoints(client):
    response = client.get("/system/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "vram_pressure" in metrics

    response = client.get("/system/tools")
    assert response.status_code == 200
    assert response.json()["function"]["name"] == "aetherforge_optimize_vram"

def test_gatekeeper_swap_matrix(client):
    # Profitable
    payload = {"mode": "high_fidelity", "estimated_context_tokens": 100, "expected_output_tokens": 2000}
    response = client.post("/system/strategy", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "strategy_applied"

    # Unprofitable
    unprofitable_payload = {"mode": "balanced", "estimated_context_tokens": 100, "expected_output_tokens": 5}
    response = client.post("/system/strategy", json=unprofitable_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

def test_thermal_circuit_breaker_enforcement(client):
    app.state.hypervisor.emergency_thermal_lock = True
    try:
        gen_response = client.post("/generate", json={"prompt": "test", "max_tokens": 10})
        assert gen_response.status_code == 503
        assert "SYSTEM LOCKED" in gen_response.json()["detail"]

        strat_response = client.post("/system/strategy", json={"mode": "high_fidelity", "expected_output_tokens": 100})
        assert strat_response.status_code == 503
    finally:
        app.state.hypervisor.emergency_thermal_lock = False

def test_failed_swap_retains_state(client):
    """Proves that a hardware-level swap failure does not mutate the API's view of the current strategy."""
    state = app.state.hypervisor
    original_strategy = state.current_strategy
    
    # Monkeypatch the engine to simulate a hard CUDA failure during swap
    original_apply = state.hardware_engine.apply_strategy
    state.hardware_engine.apply_strategy = lambda mode: {"success": False, "metrics": {}}

    try:
        payload = {"mode": "aggressive_quant", "estimated_context_tokens": 0, "expected_output_tokens": 5000}
        response = client.post("/system/strategy", json=payload)
        
        assert response.status_code == 500
        assert state.current_strategy == original_strategy, "FATAL: State mutated despite hardware swap failure."
    finally:
        # Restore mock engine
        state.hardware_engine.apply_strategy = original_apply

def test_queue_ceiling_enforcement(client):
    """Proves the Semaphore cleanly rejects traffic with 503 when the queue is saturated."""
    state = app.state.hypervisor
    
    # Artificially lock the semaphore to simulate a full queue
    async def drain_semaphore():
        for _ in range(state.semaphore._value):
            await state.semaphore.acquire()
            
    asyncio.run(drain_semaphore())

    try:
        gen_response = client.post("/generate", json={"prompt": "test"})
        assert gen_response.status_code == 503
        assert "SYSTEM BUSY" in gen_response.json()["detail"]
        
        strat_response = client.post("/system/strategy", json={"mode": "high_fidelity"})
        assert strat_response.status_code == 503
        assert "SYSTEM BUSY" in strat_response.json()["detail"]
    finally:
        # Release the locks for subsequent tests
        for _ in range(app.state.hypervisor.semaphore._value):
            app.state.hypervisor.semaphore.release()