import json
import os
from dataclasses import dataclass
from gguf import GGUFReader

@dataclass
class MoETopology:
    architecture: str
    is_moe: bool
    total_layers: int
    total_experts: int = 0
    experts_used_per_token: int = 0
    context_length: int = 0
    
    def to_dict(self):
        return self.__dict__

class GGUFAnalyzer:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot find GGUF model at: {file_path}")
        self.file_path = file_path
        self.reader = GGUFReader(self.file_path)

    def extract_topology(self) -> MoETopology:
        """
        Rips through the GGUF key-value headers to map the network structure
        without loading the massive weight tensors into RAM.
        """
        fields = self.reader.fields
        
        # 1. Identify the architecture (e.g., 'mixtral', 'qwen2moe', 'llama')
        arch_key = "general.architecture"
        if arch_key not in fields:
            raise ValueError("Invalid GGUF: Missing general.architecture key.")
        
        arch = self.reader.get_field(arch_key).parts[self.reader.get_field(arch_key).data[0]]
        arch_str = str(arch, encoding='utf-8') if isinstance(arch, bytes) else str(arch)

        # 2. Extract block count (total layers)
        layers_key = f"{arch_str}.block_count"
        total_layers = int(self.reader.get_field(layers_key).parts[self.reader.get_field(layers_key).data[0]])

        # 3. Check for MoE specific keys
        expert_count_key = f"{arch_str}.expert_count"
        expert_used_key = f"{arch_str}.expert_used_count"
        ctx_length_key = f"{arch_str}.context_length"
        
        ctx_length = 0
        if ctx_length_key in fields:
            ctx_length = int(self.reader.get_field(ctx_length_key).parts[self.reader.get_field(ctx_length_key).data[0]])

        is_moe = expert_count_key in fields
        
        if is_moe:
            total_experts = int(self.reader.get_field(expert_count_key).parts[self.reader.get_field(expert_count_key).data[0]])
            experts_used = int(self.reader.get_field(expert_used_key).parts[self.reader.get_field(expert_used_key).data[0]])
            
            return MoETopology(
                architecture=arch_str,
                is_moe=True,
                total_layers=total_layers,
                total_experts=total_experts,
                experts_used_per_token=experts_used,
                context_length=ctx_length
            )
        else:
            # Fallback for dense models (AetherForge MVP targets MoE, so we flag this)
            return MoETopology(
                architecture=arch_str,
                is_moe=False,
                total_layers=total_layers,
                context_length=ctx_length
            )

    def print_report(self, topology: MoETopology):
        print(f"\n--- AetherForge Model Recon Report ---")
        print(f"File: {os.path.basename(self.file_path)}")
        print(f"Architecture: {topology.architecture}")
        print(f"Model Type: {'Mixture of Experts (MoE)' if topology.is_moe else 'Dense Model (Unsupported for MVP)'}")
        print(f"Total Layers: {topology.total_layers}")
        
        if topology.is_moe:
            print(f"Total Experts per Layer: {topology.total_experts}")
            print(f"Active Experts per Token: {topology.experts_used_per_token}")
            print(f"Total Network Experts: {topology.total_layers * topology.total_experts}")
        print("-" * 38 + "\n")

# --- EXECUTION ---
if __name__ == "__main__":
    print("Initializing AetherForge GGUF Introspection Layer...")
    
    # Point this to a real GGUF file on your system if you have one downloaded.
    # Otherwise, this script is ready to go once you fetch DeepSeek or Mixtral.
    test_path = "../models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
    
    if os.path.exists(test_path):
        analyzer = GGUFAnalyzer(test_path)
        topo = analyzer.extract_topology()
        analyzer.print_report(topo)
        
        # Save mapping for the Cache Simulator
        with open("moe_topology.json", "w") as f:
            json.dump(topo.to_dict(), f, indent=4)
        print("Topology map exported to moe_topology.json")
    else:
        print(f"[!] Target model not found at {test_path}.")
        print("    Run this script again once you have downloaded a GGUF MoE model.")