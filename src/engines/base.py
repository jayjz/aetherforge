"""
AetherForge Base Engine Interface
=================================
Defines the strict contract for all physical and simulated backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAetherEngine(ABC):
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Must return exact token count for Gatekeeper math."""
        pass

    @abstractmethod
    def apply_strategy(self, mode: str) -> Dict[str, Any]:
        """
        Must execute hardware realignment and return exact timing metrics.
        Expected shape: {"success": bool, "metrics": {"extract_seconds": float, "reload_seconds": float, "inject_seconds": float}}
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Must yield text and execution telemetry.
        Expected shape: {"text": str, "metrics": {"tokens_generated": int, "time_seconds": float, "tokens_per_second": float}}
        """
        pass