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

    def _get_val(self, key: str):
        """Safely extracts and decodes a value from the GGUF headers."""
        field = self.reader.get_field(key)
        if not field: 
            return None
        
        try:
            # The modern gguf package method
            if hasattr(field, "contents"):
                val = field.contents()
                # Unpack single-item lists/arrays
                if isinstance(val, (list, tuple)) and len(val) == 1:
                    val = val[0]
                # Convert numpy scalars to native python types
                if hasattr(val, "item"):
                    val = val.item()
                if isinstance(val, (bytes, bytearray)):
                    return val.decode("utf-8")
                return val
            
            # The legacy gguf package fallback
            part = field.parts[field.data[0]]
            if hasattr(part, "item"):
                return part.item()
            if isinstance(part, (list, tuple)) and len(part) == 1:
                part = part[0]
            if isinstance(part, bytes):
                return part.decode('utf-8')
            return part
        except Exception:
            return None

    def _find_key_ending_with(self, suffix: str):
        """Searches all metadata keys if the standard architecture prefix fails."""
        for k in self.reader.fields.keys():
            if k.endswith(suffix):
                return self._get_val(k)
        return None

    def extract_topology(self) -> MoETopology:
        """Rips through the GGUF key-value headers safely."""
        # 1. Identify the architecture
        arch_str = self._get_val("general.architecture")
        if not arch_str:
            raise ValueError("Invalid GGUF: Missing general.architecture key.")
        arch_str = str(arch_str)

        # 2. Extract block count (total layers)
        total_layers = self._get_val(f"{arch_str}.block_count")
        if total_layers is None:
            total_layers = self._find_key_ending_with(".block_count")
        total_layers = int(total_layers) if total_layers else 0

        # 3. Check for MoE specific keys
        expert_count = self._get_val(f"{arch_str}.expert_count")
        if expert_count is None:
            expert_count = self._find_key_ending_with(".expert_count")

        experts_used = self._get_val(f"{arch_str}.expert_used_count")
        if experts_used is None:
            experts_used = self._find_key_ending_with(".expert_used_count")
            
        ctx_length = self._get_val(f"{arch_str}.context_length")
        if ctx_length is None:
            ctx_length = self._find_key_ending_with(".context_length")

        is_moe = expert_count is not None
        
        if is_moe:
            return MoETopology(
                architecture=arch_str,
                is_moe=True,
                total_layers=total_layers,
                total_experts=int(expert_count),
                experts_used_per_token=int(experts_used) if experts_used else 0,
                context_length=int(ctx_length) if ctx_length else 0
            )
        else:
            return MoETopology(
                architecture=arch_str,
                is_moe=False,
                total_layers=total_layers,
                context_length=int(ctx_length) if ctx_length else 0
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
    
    test_path = r"models\DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
    
    if os.path.exists(test_path):
        analyzer = GGUFAnalyzer(test_path)
        topo = analyzer.extract_topology()
        analyzer.print_report(topo)
        
        with open("moe_topology.json", "w") as f:
            json.dump(topo.to_dict(), f, indent=4)
        print("Topology map exported to moe_topology.json")
    else:
        print(f"[!] Target model not found at {test_path}.")