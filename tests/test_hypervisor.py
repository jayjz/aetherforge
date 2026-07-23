"""
AetherForge Hypervisor Integration Tests
========================================
Executes pure logic validation against the control plane routes using
the MockAetherEngine fallback profile. Ensures deterministic testing
by forcing headless mock variables.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

# 1. Path injection so Pytest can locate the 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Force strict testing environment BEFORE importing server logic
os.environ["AETHER_ENGINE"] = "mock"
os.environ["AETHER_CHAOS"] = "false"  # Disable random 503 lockouts during unit tests

from src.server import app


# 3. Fixture ensures the FastAPI lifespan (engine boot, topology load) runs cleanly
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_system_discovery_endpoints(client):
    """Validates that agent introspection routes return properly structured contracts."""
    # Test Cache Status
    response = client.get("/system/cache")
    assert response.status_code == 200
    data = response.json()
    assert "active_strategy" in data
    assert data["engine_available"] is False  # Headless test verification

    # Test Metrics Open Observability
    response = client.get("/system/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "vram_pressure" in metrics
    assert "performance_baselines" in metrics
    assert "thermal_lock_active" in metrics

    # Test OpenAI Function Schema Generation
    response = client.get("/system/tools")
    assert response.status_code == 200
    tools = response.json()
    assert tools["type"] == "function"
    assert tools["function"]["name"] == "aetherforge_optimize_vram"


def test_gatekeeper_swap_matrix(client):
    """Verifies that the Gatekeeper rejects unprofitable swaps and allows profitable ones."""
    # 1. Force a heavy generation payload that makes a high_fidelity swap profitable
    payload = {
        "mode": "high_fidelity",
        "estimated_context_tokens": 100,
        "expected_output_tokens": 2000
    }
    response = client.post("/system/strategy", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "strategy_applied"

    # 2. A short output request should be rejected due to swap latency overhead
    unprofitable_payload = {
        "mode": "balanced",
        "estimated_context_tokens": 100,
        "expected_output_tokens": 5
    }
    response = client.post("/system/strategy", json=unprofitable_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_thermal_circuit_breaker_enforcement(client):
    """
    Forces a thermal lock at the hypervisor state level to guarantee
    both /generate and /system/strategy endpoints return 503s.
    """
    # 1. Manually trip the emergency thermal lock
    app.state.hypervisor.emergency_thermal_lock = True

    try:
        # 2. Assert Generation path is blocked
        gen_payload = {"prompt": "test", "max_tokens": 10}
        gen_response = client.post("/generate", json=gen_payload)
        assert gen_response.status_code == 503, "Failed to block generation during thermal event"
        assert "SYSTEM LOCKED" in gen_response.json()["detail"]

        # 3. Assert Strategy path is blocked
        strat_payload = {"mode": "high_fidelity", "expected_output_tokens": 100}
        strat_response = client.post("/system/strategy", json=strat_payload)
        assert strat_response.status_code == 503, "Failed to block strategy swap during thermal event"
        assert "SYSTEM LOCKED" in strat_response.json()["detail"]

    finally:
        # 4. Release the lock so subsequent tests don't fail
        app.state.hypervisor.emergency_thermal_lock = False