import random
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ExpertMetadata:
    """Tracks the state and history of a single MoE Expert."""
    expert_id: str          # e.g., "L0_E3" (Layer 0, Expert 3)
    layer_index: int
    size_mb: float          # Assumed footprint in memory
    location: str           # "VRAM", "RAM", or "DISK"
    precision: str          # "FP16" (High) or "INT4" (Low)
    last_used_step: int = 0
    use_count: int = 0
    cumulative_importance: float = 0.0

class AetherCacheManager:
    def __init__(self, vram_budget_mb: float, ram_budget_mb: float):
        self.vram_budget_mb = vram_budget_mb
        self.ram_budget_mb = ram_budget_mb
        self.experts: Dict[str, ExpertMetadata] = {}
        
        # Track current usage
        self.current_vram_usage = 0.0
        self.current_step = 0

    def register_expert(self, layer_index: int, expert_index: int, size_mb: float):
        """Initializes an expert's tracking data (starts on DISK/RAM)."""
        ext_id = f"L{layer_index}_E{expert_index}"
        self.experts[ext_id] = ExpertMetadata(
            expert_id=ext_id,
            layer_index=layer_index,
            size_mb=size_mb,
            location="RAM", # Assume default offloaded state
            precision="INT4"
        )

    def _calculate_retention_score(self, expert: ExpertMetadata) -> float:
        """
        The HOBBIT algorithm: Blends Frequency (LFU), Recency (LRU), and Importance.
        High score = keep in VRAM. Low score = evict to RAM.
        """
        if expert.use_count == 0:
            return 0.0
            
        recency_penalty = self.current_step - expert.last_used_step
        
        # Weights can be tuned based on workload profiling
        weight_importance = 0.5
        weight_frequency = 0.3
        weight_recency = -0.2 
        
        score = (
            (expert.cumulative_importance * weight_importance) +
            (expert.use_count * weight_frequency) +
            (recency_penalty * weight_recency)
        )
        return score

    def _evict_until_space_available(self, required_mb: float):
        """Finds the lowest scoring experts in VRAM and demotes them to RAM."""
        vram_experts = [e for e in self.experts.values() if e.location == "VRAM"]
        
        while self.current_vram_usage + required_mb > self.vram_budget_mb:
            if not vram_experts:
                raise MemoryError("VRAM budget too small to hold required experts!")
                
            # Sort by lowest retention score
            vram_experts.sort(key=self._calculate_retention_score)
            
            # Evict the weakest link
            weakest_expert = vram_experts.pop(0)
            weakest_expert.location = "RAM"
            weakest_expert.precision = "INT4"
            self.current_vram_usage -= weakest_expert.size_mb
            
            print(f"  [Evicting] {weakest_expert.expert_id} to RAM (Score: {self._calculate_retention_score(weakest_expert):.2f})")

    def route_token(self, layer_index: int, active_experts: List[tuple[int, float]]):
        """
        Simulates the MoE router selecting experts for a token.
        active_experts: List of (expert_index, router_probability)
        """
        self.current_step += 1
        
        for exp_idx, router_prob in active_experts:
            ext_id = f"L{layer_index}_E{exp_idx}"
            expert = self.experts[ext_id]
            
            # Update metrics
            expert.use_count += 1
            expert.last_used_step = self.current_step
            expert.cumulative_importance += router_prob
            
            # HOBBIT Logic: If importance is high enough, we want it in VRAM at FP16
            # For MVP simulation, we try to put all routed experts in VRAM
            if expert.location != "VRAM":
                print(f"  [Cache Miss] {ext_id} not in VRAM. Fetching...")
                self._evict_until_space_available(expert.size_mb)
                
                expert.location = "VRAM"
                expert.precision = "FP16" if router_prob > 0.4 else "INT4" # Mixed precision threshold
                self.current_vram_usage += expert.size_mb
            else:
                print(f"  [Cache Hit] {ext_id} already in VRAM.")

    def print_vram_state(self):
        vram_list = [e.expert_id for e in self.experts.values() if e.location == "VRAM"]
        print(f"\n--- Step {self.current_step} | VRAM Usage: {self.current_vram_usage}MB / {self.vram_budget_mb}MB ---")
        print(f"Experts in VRAM: {', '.join(vram_list)}\n")


# --- SIMULATION WORKLOAD ---
if __name__ == "__main__":
    print("Initializing AetherForge Cache Simulator...")
    
    # Simulate a small 8GB VRAM constraint (8000 MB)
    # Assume we have 4 layers, 8 experts per layer, each expert is 1000MB
    manager = AetherCacheManager(vram_budget_mb=4000, ram_budget_mb=32000)
    
    for l in range(4):
        for e in range(8):
            manager.register_expert(layer_index=l, expert_index=e, size_mb=1000.0)
            
    print("Running token generation simulation...\n")
    
    # Simulate 10 tokens passing through Layer 0
    # Watch how it realizes Experts 1 and 2 are used heavily and locks them in,
    # while evicting random ones.
    for step in range(1, 11):
        print(f"Token {step} entering Layer 0:")
        
        # Fake router output: Expert 1 is heavily favored, Expert 2 is secondary, 
        # plus one random expert to cause cache thrashing
        active = [
            (1, 0.65), # High importance
            (2, 0.25), # Med importance
            (random.randint(3, 7), 0.10) # Low importance noise
        ]
        
        manager.route_token(layer_index=0, active_experts=active)
        manager.print_vram_state()