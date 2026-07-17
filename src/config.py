"""
AetherForge Configuration State
===============================
Single source of truth for all hardware limits, file paths, and Gatekeeper 
heuristics. Automatically parses from environment variables or a local .env file.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class AetherSettings(BaseSettings):
    # Hardware & Model Configuration
    model_path: str = Field(
        default=os.path.join("models", "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"), 
        description="Path to the GGML/GGUF model file."
    )
    vram_budget_mb: int = Field(default=8000, gt=0, description="Target VRAM limit in MB.")
    ram_budget_mb: int = Field(default=32000, gt=0, description="Target System RAM limit in MB.")
    n_ctx: int = Field(default=4096, gt=0, description="Context window size.")

    # Strategy To GPU Layer Mapping 
    # (Defaults calibrated for RTX 4060 8GB + DeepSeek-Lite)
    layers_high_fidelity: int = Field(default=15, ge=0, description="GPU Layers allocated during high_fidelity mode.")
    layers_balanced: int = Field(default=10, ge=0, description="GPU Layers allocated during balanced mode.")
    layers_aggressive_quant: int = Field(default=2, ge=0, description="GPU Layers allocated during aggressive_quant mode.")

    # Gatekeeper Tuning & Heuristics
    swap_penalty_seconds: float = Field(default=5.8, ge=0.0, description="Base time in seconds to perform a Fast-Swap.")
    state_io_base_seconds: float = Field(default=0.5, ge=0.0, description="Base IO overhead for memory serialization.")
    state_io_per_token_seconds: float = Field(default=0.0001, ge=0.0, description="RAM IO scaling factor per token.")
    # Gatekeeper Live Telemetry
    telemetry_alpha: float = Field(default=0.3, ge=0.0, le=1.0, description="EMA smoothing factor.")
    tps_min_clamp: float = Field(default=2.0, description="Absolute minimum TPS floor to prevent death spirals.")
    tps_max_clamp: float = Field(default=60.0, description="Absolute maximum TPS ceiling.")

    tps_high_fidelity: float = Field(default=23.71, gt=0.0, description="Estimated decode speed for high_fidelity.")
    tps_balanced: float = Field(default=11.10, gt=0.0, description="Estimated decode speed for balanced.")
    tps_aggressive_quant: float = Field(default=12.10, gt=0.0, description="Estimated decode speed for aggressive_quant.")

    # Server Configuration
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, gt=0, le=65535)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = AetherSettings()