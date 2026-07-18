"""
AetherForge Hypervisor Integration Tests
========================================
Executes pure logic validation against the control plane routes using
the MockAetherEngine fallback profile.
"""

import pytest
from fastapi.testclient import TestClient
from src.server import app
from src.config import settings

client = TestClient(app)

def test_system_discovery_endpoints():
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

    # Test OpenAI Function Schema Generation
    response = client.get("/system/tools")
    assert response.status_code == 200
    tools = response.json()
    assert tools["type"] == "function"
    assert tools["function"]["name"] == "aetherforge_optimize_vram"


def test_gatekeeper_swap_matrix():
    """Verifies that the Gatekeeper rejects unprofitable swaps and allows profitable ones."""
    
    # Force a heavy generation payload that makes a high_fidelity swap profitable
    payload = {
        "mode": "high_fidelity",
        "estimated_context_tokens": 100,
        "expected_output_tokens": 2000  # High token count justifies the swap penalty
    }
    response = client.post("/system/strategy", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "simulation_strategy_accepted"

    # A short output request should be rejected due to swap latency overhead
    unprofitable_payload = {
        "mode": "balanced",
        "estimated_context_tokens": 100,
        "expected_output_tokens": 5  # Too small to be worth a swap penalty
    }
    response = client.post("/system/strategy", json=unprofitable_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"