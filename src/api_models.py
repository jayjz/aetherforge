from pydantic import BaseModel, Field
from typing import List, Optional

class HypervisorStrategy(BaseModel):
    """Allows an agent to dynamically alter memory behavior."""
    mode: str = Field(..., description="E.g., 'high_fidelity', 'aggressive_quant', 'balanced'")
    priority_layers: Optional[List[int]] = Field(None, description="Force specific layers to stay in VRAM")

class InferenceRequest(BaseModel):
    """The payload an agent sends to generate text."""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    strategy: Optional[HypervisorStrategy] = None

class ExpertState(BaseModel):
    expert_id: str
    location: str
    precision: str

class CacheStatusResponse(BaseModel):
    """What the hypervisor reports back to the agent."""
    vram_budget_mb: float
    current_vram_usage_mb: float
    active_experts_in_vram: List[ExpertState]
    status: str