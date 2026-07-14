"""
AetherForge Agent Integration Tool
==================================
Provides a standardized interface for autonomous agents to command 
the AetherForge Hypervisor via REST API. Designed to be drop-in 
compatible with OpenAI Function Calling, LangChain, and custom loops.
"""

import requests
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

# Setup basic logging for the agent environment
logging.basicConfig(level=logging.INFO, format="[Agent Tool] %(message)s")
logger = logging.getLogger(__name__)

class StrategyConfig(BaseModel):
    """Schema for validating agent strategy commands."""
    mode: str = Field(
        ..., 
        description="The VRAM allocation mode. Must be 'high_fidelity' (coding/reasoning), 'aggressive_quant' (summarization), or 'balanced'."
    )
    priority_layers: Optional[List[int]] = Field(
        default_factory=list, 
        description="Specific MoE layers to lock into VRAM. Leave empty unless specifically targeted."
    )

class AetherForgeTool:
    """
    The client library agents use to command the local hardware hypervisor.
    """
    
    def __init__(self, api_base_url: str = "http://127.0.0.1:8000"):
        self.api_base_url = api_base_url.rstrip("/")
        self.session = requests.Session()
        
    def check_system_status(self) -> Dict[str, Any]:
        """Queries the current VRAM usage and active memory strategy."""
        try:
            response = self.session.get(f"{self.api_base_url}/system/cache", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Hypervisor connection failed: {e}")
            return {"status": "offline", "error": str(e)}

    def inject_strategy(self, mode: str, priority_layers: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Commands the hypervisor to shift its VRAM allocation strategy.
        """
        # Validate through Pydantic
        config = StrategyConfig(mode=mode, priority_layers=priority_layers)
        
        try:
            logger.info(f"Injecting strategy: {config.mode.upper()}")
            response = self.session.post(
                f"{self.api_base_url}/system/strategy",
                json=config.model_dump(),
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to inject strategy: {e}")
            return {"status": "failed", "error": str(e)}

    @classmethod
    def get_openai_function_schema(cls) -> Dict[str, Any]:
        """
        Returns the exact JSON schema required to pass this tool 
        into an OpenAI-compatible agent's 'tools' array.
        """
        return {
            "type": "function",
            "function": {
                "name": "aetherforge_inject_strategy",
                "description": (
                    "Commands the local AI hardware hypervisor to change its memory "
                    "allocation strategy before executing a task. Use 'high_fidelity' "
                    "for complex reasoning/coding, and 'aggressive_quant' for simple tasks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["high_fidelity", "aggressive_quant", "balanced"],
                            "description": "The VRAM strategy to apply."
                        },
                        "priority_layers": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional specific MoE layers to lock into VRAM.",
                            "default": []
                        }
                    },
                    "required": ["mode"]
                }
            }
        }