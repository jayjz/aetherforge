"""
AetherForge Cache Manager (The Brain)
=====================================
Predictive VRAM management for Mixture of Experts (MoE) models.
Generates strict MemoryInstructions based on router probabilities.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

# Import the strict hardware contract
from src.tensor_bridge import MemoryInstruction


@dataclass(frozen=True)
class CacheWeights:
    """Hyperparameters for the predictive retention scoring algorithm."""
    importance: float = 0.5
    frequency: float = 0.3
    recency: float = -0.2


@dataclass
class ExpertMetadata:
    """Tracks the lifecycle, size, and routing history of a single MoE Expert."""
    expert_id: str
    layer_index: int
    expert_index: int
    size_mb: float
    location: str
    precision: str
    last_used_step: int = 0
    use_count: int = 0
    cumulative_importance: float = 0.0


class AetherCacheManager:
    """
    Simulates and manages physical VRAM limits, deciding which MoE experts 
    should be loaded, evicted, or retained during token generation.
    """
    
    def __init__(self, vram_budget_mb: float, ram_budget_mb: float):
        self.vram_budget_mb = vram_budget_mb
        self.ram_budget_mb = ram_budget_mb
        self.experts: Dict[str, ExpertMetadata] = {}
        
        self.current_vram_usage: float = 0.0
        self.current_step: int = 0
        
        self.weights = CacheWeights()
        self.architecture: str = "unknown"
        self.active_experts_per_token: int = 0

    def load_topology(self, topology_path: str, estimated_expert_size_mb: float = 120.0) -> None:
        """Builds the memory map dynamically from the Introspection Layer's output."""
        if not os.path.exists(topology_path):
            raise FileNotFoundError(f"Topology file missing: {topology_path}. Run model_analyzer.py first.")
            
        with open(topology_path, 'r') as f:
            topo = json.load(f)
            
        if not topo.get("is_moe", False):
            raise ValueError("AetherForge MVP strictly supports MoE models. Dense model detected.")
            
        self.architecture = topo["architecture"]
        self.active_experts_per_token = topo.get("experts_used_per_token", 2)
        total_layers = int(topo["total_layers"])
        total_experts = int(topo["total_experts"])
        
        print(f"[Brain] Registering Topology: {self.architecture.upper()}")
        print(f"[Brain] Mapping {total_layers} layers with {total_experts} experts each...")
        
        for layer in range(total_layers):
            for expert in range(total_experts):
                self._register_expert(layer, expert, estimated_expert_size_mb)
                
        print(f"[Brain] Successfully registered {len(self.experts)} total experts.")

    def _register_expert(self, layer_index: int, expert_index: int, size_mb: float) -> None:
        """Internal helper to instantiate expert tracking metadata."""
        ext_id = f"L{layer_index}_E{expert_index}"
        self.experts[ext_id] = ExpertMetadata(
            expert_id=ext_id,
            layer_index=layer_index,
            expert_index=expert_index,
            size_mb=size_mb,
            location="RAM",
            precision="INT4"
        )

    def _calculate_retention_score(self, expert: ExpertMetadata) -> float:
        """Calculates VRAM priority based on frequency, recency, and routing weight."""
        if expert.use_count == 0:
            return 0.0
            
        recency_penalty = self.current_step - expert.last_used_step
        
        return (
            (expert.cumulative_importance * self.weights.importance) +
            (expert.use_count * self.weights.frequency) +
            (recency_penalty * self.weights.recency)
        )

    def _evict_until_space_available(self, required_mb: float) -> Set[int]:
        """
        Forces lowest-scoring experts out of VRAM until the required space is freed.
        Returns a set of the expert indices that were physically evicted.
        """
        evicted_indices = set()
        vram_experts = [e for e in self.experts.values() if e.location == "VRAM"]
        
        while self.current_vram_usage + required_mb > self.vram_budget_mb:
            if not vram_experts:
                raise MemoryError("VRAM budget too small to hold required experts!")
                
            # Sort by lowest retention score first
            vram_experts.sort(key=self._calculate_retention_score)
            weakest = vram_experts.pop(0)
            
            # Update state
            weakest.location = "RAM"
            weakest.precision = "INT4"
            self.current_vram_usage -= weakest.size_mb
            evicted_indices.add(weakest.expert_index)
            
        return evicted_indices

    def route_token(self, layer_index: int, active_experts: List[Tuple[int, float]]) -> MemoryInstruction:
        """
        Core logic: Processes a token's router probabilities and generates a strict
        MemoryInstruction for the Hardware Engine to execute.
        """
        self.current_step += 1
        
        load_experts: Set[int] = set()
        evict_experts: Set[int] = set()
        retain_experts: Set[int] = set()
        
        for exp_idx, router_prob in active_experts:
            ext_id = f"L{layer_index}_E{exp_idx}"
            expert = self.experts.get(ext_id)
            
            if not expert:
                continue
            
            # Update analytical history
            expert.use_count += 1
            expert.last_used_step = self.current_step
            expert.cumulative_importance += router_prob
            
            if expert.location != "VRAM":
                # Ensure space exists, capturing any collateral evictions
                freed_indices = self._evict_until_space_available(expert.size_mb)
                evict_experts.update(freed_indices)
                
                # Execute theoretical load
                expert.location = "VRAM"
                expert.precision = "FP16" if router_prob > 0.4 else "INT4"
                self.current_vram_usage += expert.size_mb
                
                load_experts.add(expert.expert_index)
            else:
                retain_experts.add(expert.expert_index)

        # Generate the immutable hardware command
        return MemoryInstruction(
            layer_index=layer_index,
            load_experts=load_experts,
            evict_experts=evict_experts,
            retain_experts=retain_experts
        )

    def print_vram_state(self) -> None:
        """Outputs a clean terminal representation of current memory distribution."""
        vram_list = [e.expert_id for e in self.experts.values() if e.location == "VRAM"]
        print(f"\n--- Step {self.current_step} | VRAM: {self.current_vram_usage:.0f}MB / {self.vram_budget_mb}MB ---")
        print(f"Active in VRAM ({len(vram_list)} experts): {', '.join(vram_list[:10])}...")


# --- SIMULATION EXECUTION ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    topology_file = os.path.join(base_dir, "moe_topology.json")
    
    if not os.path.exists(topology_file):
        print("[!] moe_topology.json not found. Generating a mock Mixtral topology...")
        mock_topo = {
            "architecture": "mixtral", "is_moe": True, 
            "total_layers": 32, "total_experts": 8, "experts_used_per_token": 2
        }
        with open(topology_file, "w") as f:
            json.dump(mock_topo, f)

    # Initialize 8GB VRAM Simulator
    manager = AetherCacheManager(vram_budget_mb=8000, ram_budget_mb=32000)
    manager.load_topology(topology_file, estimated_expert_size_mb=120.0)
    
    print("\n[Simulation] Routing 5 tokens through Layer 0...")
    
    for step in range(1, 6):
        active = [(3, 0.70), (random.randint(0, 7), 0.25)]
        instruction = manager.route_token(layer_index=0, active_experts=active)
        
        print(f"\nToken {step} Generated Instruction:")
        print(f"  -> Load: {instruction.load_experts}")
        print(f"  -> Evict: {instruction.evict_experts}")
        print(f"  -> Retain: {instruction.retain_experts}")
        print(f"  -> NO-OP? {instruction.is_noop()}")
        
    manager.print_vram_state()