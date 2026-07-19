"""
AetherForge Configuration State
===============================
Single source of truth for all hardware limits, file paths, and Gatekeeper 
heuristics. Dynamically merges a local config.yaml file with environment overrides.
Enforces strict fail-fast bounds to guarantee silicon safety.
"""

import os
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Dict, Any, Optional

class AetherSettings(BaseSettings):
    # Hardware & Model Configuration
    model_path: str = Field(default="models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf")
    vram_budget_mb: int = Field(default=8000, gt=0)
    ram_budget_mb: int = Field(default=32000, gt=0)
    n_ctx: int = Field(default=8192, gt=0)

    # Strategy To GPU Layer Mapping 
    layers_high_fidelity: int = Field(default=15, ge=0)
    layers_balanced: int = Field(default=10, ge=0)
    layers_aggressive_quant: int = Field(default=2, ge=0)

    # Gatekeeper Tuning & Heuristics
    swap_penalty_seconds: float = Field(default=5.8, ge=0.0)
    state_io_base_seconds: float = Field(default=0.5, ge=0.0)
    state_io_per_token_seconds: float = Field(default=0.0001, ge=0.0)
    
    # Gatekeeper Live Telemetry & Clamping
    telemetry_alpha: float = Field(default=0.3, ge=0.01, le=1.0)
    tps_min_clamp: float = Field(default=2.0, ge=0.1)
    tps_max_clamp: float = Field(default=60.0, le=200.0)

    # --- ABSOLUTE HARDWARE KILL-SWITCH BOUNDS ---
    # These limits are hard-coded to protect the 8GB RTX 4060. 
    # Any yaml override outside these bounds will fatally crash the app on boot.
    max_safe_context_tokens: int = Field(default=8192, le=32000)
    max_gpu_temp_c: int = Field(default=75, ge=60, le=88)
    max_vram_allocation_pct: float = Field(default=85.0, ge=50.0, le=98.0)

    # Performance Estimates
    tps_high_fidelity: float = Field(default=23.71, gt=0)
    tps_balanced: float = Field(default=11.10, gt=0)
    tps_aggressive_quant: float = Field(default=12.10, gt=0)

    # Server Configuration
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, gt=0, le=65535)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "config.yaml") -> "AetherSettings":
        """Pre-parses a YAML file to populate settings, enforcing fail-fast validation."""
        if not os.path.exists(yaml_path):
            print(f"[Config] No {yaml_path} detected. Initializing standard environment state.")
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f) or {}

        # Flatten nested YAML structure into flat Pydantic attributes
        flat_data = {}
        if "model" in raw_yaml:
            flat_data["model_path"] = raw_yaml["model"].get("path")
            flat_data["n_ctx"] = raw_yaml["model"].get("n_ctx")
            
        if "strategies" in raw_yaml:
            for mode, profile in raw_yaml["strategies"].items():
                flat_data[f"layers_{mode}"] = profile.get("gpu_layers")
                flat_data[f"tps_{mode}"] = profile.get("tps_estimate")

        if "gatekeeper" in raw_yaml:
            gk = raw_yaml["gatekeeper"]
            flat_data["swap_penalty_seconds"] = gk.get("swap_penalty_seconds")
            flat_data["telemetry_alpha"] = gk.get("telemetry_alpha")
            flat_data["max_gpu_temp_c"] = gk.get("max_gpu_temp_c")
            flat_data["max_vram_allocation_pct"] = gk.get("max_vram_pct")
            flat_data["max_safe_context_tokens"] = gk.get("max_safe_context_tokens")

        if "server" in raw_yaml:
            flat_data["api_host"] = raw_yaml["server"].get("host")
            flat_data["api_port"] = raw_yaml["server"].get("port")

        # Strip out None values to allow fields to fall back to Pydantic defaults
        cleaned_data = {k: v for k, v in flat_data.items() if v is not None}
        
        print(f"[Config] Successfully compiled state from local {yaml_path}")
        return cls(**cleaned_data)

# Global instantiation hook
settings = AetherSettings.load_from_yaml()