"""
AetherForge Engine Abstraction Test
===================================
Validates that the Engine Factory cleanly loads the requested backend
and that the returned object strictly adheres to the BaseAetherEngine contract.
"""

import sys
from src.engines import create_engine
from src.engines.base import BaseAetherEngine

def run_abstraction_tests():
    print("🧪 INITIATING ENGINE ABSTRACTION TESTS")
    print("======================================")

    try:
        # 1. Test Factory Loading
        print(" -> Testing Factory initialization (Mock Engine)...")
        engine = create_engine(
            engine_type="mock",
            model_path="dummy/path/model.gguf",
            vram_budget_mb=8000,
            n_ctx=4096
        )
        
        assert isinstance(engine, BaseAetherEngine), "Engine does not inherit from BaseAetherEngine contract!"
        print(" -> [PASS] Engine successfully instantiated and matches BaseAetherEngine type.")

        # 2. Test count_tokens Contract
        print("\n -> Testing count_tokens contract...")
        dummy_text = "The hypervisor must maintain absolute state coherence."
        token_count = engine.count_tokens(dummy_text)
        assert isinstance(token_count, int) and token_count > 0, f"Invalid token count: {token_count}"
        print(f" -> [PASS] count_tokens returned valid integer: {token_count}")

        # 3. Test apply_strategy Contract
        print("\n -> Testing apply_strategy contract...")
        strategy_result = engine.apply_strategy("high_fidelity")
        
        assert isinstance(strategy_result, dict), "apply_strategy must return a dictionary."
        assert "success" in strategy_result and isinstance(strategy_result["success"], bool), "Missing or invalid 'success' key."
        assert "metrics" in strategy_result and isinstance(strategy_result["metrics"], dict), "Missing or invalid 'metrics' dict."
        
        metrics = strategy_result["metrics"]
        assert "extract_seconds" in metrics and isinstance(metrics["extract_seconds"], float), "Missing/invalid 'extract_seconds'."
        assert "reload_seconds" in metrics and isinstance(metrics["reload_seconds"], float), "Missing/invalid 'reload_seconds'."
        assert "inject_seconds" in metrics and isinstance(metrics["inject_seconds"], float), "Missing/invalid 'inject_seconds'."
        print(" -> [PASS] apply_strategy returned perfect dictionary shape.")

        # 4. Test generate Contract
        print("\n -> Testing generate contract...")
        gen_result = engine.generate(prompt="Test prompt", max_tokens=10)
        
        assert isinstance(gen_result, dict), "generate must return a dictionary."
        assert "text" in gen_result and isinstance(gen_result["text"], str), "Missing or invalid 'text' key."
        assert "metrics" in gen_result and isinstance(gen_result["metrics"], dict), "Missing or invalid 'metrics' dict."
        
        gen_metrics = gen_result["metrics"]
        assert "tokens_generated" in gen_metrics and isinstance(gen_metrics["tokens_generated"], int), "Missing/invalid 'tokens_generated'."
        assert "time_seconds" in gen_metrics and isinstance(gen_metrics["time_seconds"], float), "Missing/invalid 'time_seconds'."
        assert "tokens_per_second" in gen_metrics and isinstance(gen_metrics["tokens_per_second"], (float, int)), "Missing/invalid 'tokens_per_second'."
        print(" -> [PASS] generate returned perfect dictionary shape.")

        print("\n✅ ALL ABSTRACTION CONTRACT TESTS PASSED.")
        
    except AssertionError as e:
        print(f"\n❌ [FAIL] Contract violation detected: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ [FATAL] Unexpected error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_abstraction_tests()