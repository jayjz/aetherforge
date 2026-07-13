import os
import time
from llama_cpp import Llama
from src.cache_manager import AetherCacheManager

class AetherEngine:
    def __init__(self, model_path: str, vram_budget_mb: float = 8000):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model_path = model_path
        self.cache_manager = AetherCacheManager(vram_budget_mb=vram_budget_mb, ram_budget_mb=32000)
        
        print(f"[Engine] Booting AetherForge Inference Core...")
        print(f"[Engine] Target Model: {os.path.basename(model_path)}")
        
        # Initialize llama.cpp
        # n_gpu_layers=10 is a conservative start for 8GB VRAM with a 16B MoE model
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=10, 
            n_ctx=2048,
            verbose=False
        )
        print("[Engine] CUDA Engine Online.")

    def generate(self, prompt: str, max_tokens: int = 100):
        print(f"\n[Engine] Received Prompt: '{prompt}'")
        print("[Engine] Commencing generation...\n")
        
        start_time = time.time()
        
        # Standard generation call
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            echo=False
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        tokens_generated = output['usage']['completion_tokens']
        tps = tokens_generated / generation_time
        
        print(f"\nResult:\n{output['choices'][0]['text']}")
        print(f"\n--- AetherForge Baseline Metrics ---")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Time taken: {generation_time:.2f} seconds")
        print(f"Speed: {tps:.2f} tokens/sec")

if __name__ == "__main__":
    # Point to the DeepSeek model we just downloaded
    target_model = r"models\DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
    
    engine = AetherEngine(model_path=target_model, vram_budget_mb=8000)
    
    test_prompt = "Write a highly optimized Python function to calculate the Fibonacci sequence."
    engine.generate(prompt=test_prompt, max_tokens=150)