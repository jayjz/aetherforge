import random
import json
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ExpertMetadata:
    """Tracks the state and history of a single MoE Expert."""
    expert_id: str
    layer_index: int
    size_mb: float
    location: str
    precision: str
    last_used_step: int = 0
    use_count: int = 0
    cumulative_importance: float = 0.0

class AetherCacheManager:
    def __init__(self, vram_budget_mb: float, ram_budget_mb: float):
        self.vram_budget_mb = vram_budget_mb
        self.ram_budget_mb = ram_budget_mb
        self.experts: Dict[str, ExpertMetadata] = {}
        
        self.current_vram_usage = 0.0
        self.current_step = 0
        
        # Topology metadata
        self.architecture = "unknown"
        self.active_experts_per_token = 0

    def load_topology(self, topology_path: str, estimated_expert_size_mb: float = 120.0):
        """Builds the memory map dynamically from the Introspection Layer's output."""
        if not os.path.exists(topology_path):
            raise FileNotFoundError(f"Topology file missing: {topology_path}. Run model_analyzer.py first.")
            
        with open(topology_path, 'r') as f:
            topo = json.load(f)
            
        if not topo.get("is_moe", False):
            raise ValueError("AetherForge MVP strictly supports MoE models. Dense model detected.")
            
        self.architecture = topo["architecture"]
        self.active_experts_per_token = topo.get("experts_used_per_token", 2)
        total_layers = topo["total_layers"]
        total_experts = topo["total_experts"]
        
        print(f"[Brain] Registering Topology: {self.architecture.upper()}")
        print(f"[Brain] Mapping {total_layers} layers with {total_experts} experts each...")
        
        for l in range(total_layers):
            for e in range(total_experts):
                self.register_expert(layer_index=l, expert_index=e, size_mb=estimated_expert_size_mb)
                
        print(f"[Brain] Successfully registered {len(self.experts)} total experts.")

    def register_expert(self, layer_index: int, expert_index: int, size_mb: float):
        ext_id = f"L{layer_index}_E{expert_index}"
        self.experts[ext_id] = ExpertMetadata(
            expert_id=ext_id,
            layer_index=layer_index,
            size_mb=size_mb,
            location="RAM",
            precision="INT4"
        )

    def _calculate_retention_score(self, expert: ExpertMetadata) -> float:
        if expert.use_count == 0:
            return 0.0
            
        recency_penalty = self.current_step - expert.last_used_step
        
        weight_importance = 0.5
        weight_frequency = 0.3
        weight_recency = -0.2 
        
        return (
            (expert.cumulative_importance * weight_importance) +
            (expert.use_count * weight_frequency) +
            (recency_penalty * weight_recency)
        )

    def _evict_until_space_available(self, required_mb: float):
        vram_experts = [e for e in self.experts.values() if e.location == "VRAM"]
        
        while self.current_vram_usage + required_mb > self.vram_budget_mb:
            if not vram_experts:
                raise MemoryError("VRAM budget too small to hold required experts!")
                
            vram_experts.sort(key=self._calculate_retention_score)
            weakest = vram_experts.pop(0)
            
            weakest.location = "RAM"
            weakest.precision = "INT4"
            self.current_vram_usage -= weakest.size_mb
            
            # Print commented out to reduce noise in large simulations
            # print(f"  [Evict] {weakest.expert_id} -> RAM")

    def route_token(self, layer_index: int, active_experts: List[tuple[int, float]]):
        self.current_step += 1
        
        for exp_idx, router_prob in active_experts:
            ext_id = f"L{layer_index}_E{exp_idx}"
            expert = self.experts.get(ext_id)
            if not expert:
                continue
            
            expert.use_count += 1
            expert.last_used_step = self.current_step
            expert.cumulative_importance += router_prob
            
            if expert.location != "VRAM":
                self._evict_until_space_available(expert.size_mb)
                expert.location = "VRAM"
                expert.precision = "FP16" if router_prob > 0.4 else "INT4"
                self.current_vram_usage += expert.size_mb

    def print_vram_state(self):
        vram_list = [e.expert_id for e in self.experts.values() if e.location == "VRAM"]
        print(f"\n--- Step {self.current_step} | VRAM: {self.current_vram_usage:.0f}MB / {self.vram_budget_mb}MB ---")
        print(f"Active in VRAM ({len(vram_list)} experts): {', '.join(vram_list[:10])}... [truncated]")


# --- SIMULATION EXECUTION ---
if __name__ == "__main__":
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    topology_file = os.path.join(base_dir, "moe_topology.json")
    
    # Generate a mock topology if one doesn't exist, so you can test immediately
    if not os.path.exists(topology_file):
        print("[!] moe_topology.json not found. Generating a mock Mixtral 8x7B topology...")
        mock_topo = {
            "architecture": "mixtral", "is_moe": True, 
            "total_layers": 32, "total_experts": 8, "experts_used_per_token": 2
        }
        with open(topology_file, "w") as f:
            json.dump(mock_topo, f)

    # Initialize 8GB VRAM Simulator
    manager = AetherCacheManager(vram_budget_mb=8000, ram_budget_mb=32000)
    
    # Wire the Eyes to the Brain
    # Using 120MB per expert (Roughly accurate for Q4_K_M Mixtral)
    manager.load_topology(topology_file, estimated_expert_size_mb=120.0)
    
    print("\n[Simulation] Routing 5 tokens through Layer 0...")
    
    # Simulate routing with realistic Mixtral behavior (2 active experts per token)
    for step in range(1, 6):
        # Simulate heavy preference for Expert 3, varying secondary experts
        active = [(3, 0.70), (random.randint(0, 7), 0.25)]
        manager.route_token(layer_index=0, active_experts=active)
        
    manager.print_vram_state()