"""Pydantic-based configuration with YAML loading and environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ── Sub-models ──────────────────────────────────────────────────────────────

class GeneralConfig(BaseModel):
    project_name: str = "ekap-anom"
    env: str = "dev"
    log_level: str = "INFO"
    log_format: str = "json"
    data_root: str = "./data"
    model_root: str = "./models"


class IngestConfig(BaseModel):
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    chunk_size: int = 100_000
    compression: str = "snappy"


class FilterConfig(BaseModel):
    dynamic_extensions: list[str] = Field(default_factory=lambda: ["aspx", "ashx"])
    optional_extensions: list[str] = Field(default_factory=lambda: ["asmx", "svc"])
    enable_optional: bool = False

    @property
    def allowed_extensions(self) -> set[str]:
        exts = set(self.dynamic_extensions)
        if self.enable_optional:
            exts |= set(self.optional_extensions)
        return exts


class SegmentationConfig(BaseModel):
    enable_host: bool = False
    enable_bot_flag: bool = True
    min_segment_count: int = 5000


class URLConfig(BaseModel):
    lowercase: bool = True
    strip_trailing_slash: bool = True
    numeric_template: bool = True
    aggressive_template: bool = False


class UAConfig(BaseModel):
    bot_patterns: list[str] = Field(default_factory=lambda: [
        r"(?i)bot", r"(?i)crawler", r"(?i)spider", r"(?i)slurp",
        r"(?i)wget", r"(?i)curl", r"(?i)python-requests", r"(?i)httpx", r"(?i)scrapy",
    ])
    novelty_window_days: int = 7


class HashingConfig(BaseModel):
    path_hash_dim: int = 256
    query_hash_dim: int = 128


class Layer1Config(BaseModel):
    window_size_minutes: int = 5
    step_minutes: int = 1
    rolling_hours: int = 3
    hot_threshold: float = 0.8
    use_ewma: bool = False
    ewma_alpha: float = 0.3


class IForestConfig(BaseModel):
    n_estimators: int = 200
    max_samples: str | int = "auto"
    contamination: float = 0.01
    random_state: int = 42


class AutoencoderConfig(BaseModel):
    latent_dim: int = 32
    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    noise_factor: float = 0.1
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 512


class Layer2Config(BaseModel):
    mode: str = "iforest"  # iforest | ae_latent_if
    iforest: IForestConfig = Field(default_factory=IForestConfig)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)


class Layer3Config(BaseModel):
    enabled: bool = False
    session_timeout_minutes: int = 30
    critical_endpoints: list[str] = Field(default_factory=lambda: [
        "/EKAP/Default.aspx",
        "/EKAP/Ortak/YeniIhaleAramaData.ashx",
        "/EKAP/Teklif/IhaleDokumanDownload.aspx",
        "/EKAP/Istekli/IstekliBilgileri.ashx",
    ])


class ScoringWeights(BaseModel):
    layer1: float = 0.3
    layer2: float = 0.5
    layer3: float = 0.2


class ScoringConfig(BaseModel):
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    segment_quantile: float = 0.995
    use_evt: bool = False
    topk_daily: int = 200
    topk_critical: int = 50


class PrivacyConfig(BaseModel):
    anonymize_ip: bool = True
    hash_cookie: bool = True
    redact_query_values: bool = True


class DriftConfig(BaseModel):
    psi_threshold: float = 0.2
    jsd_threshold: float = 0.1
    score_shift_threshold: float = 0.15
    retrain_schedule: str = "weekly"


class MLflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "ekap-anomaly"
    registry_name: str = "ekap-anom-models"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2


# ── Root Config ─────────────────────────────────────────────────────────────

class Settings(BaseModel):
    """Root configuration assembled from YAML + env overrides."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    url: URLConfig = Field(default_factory=URLConfig)
    ua: UAConfig = Field(default_factory=UAConfig)
    hashing: HashingConfig = Field(default_factory=HashingConfig)
    layer1: Layer1Config = Field(default_factory=Layer1Config)
    layer2: Layer2Config = Field(default_factory=Layer2Config)
    layer3: Layer3Config = Field(default_factory=Layer3Config)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    critical_endpoints: list[str] = Field(default_factory=lambda: [
        "/EKAP/Default.aspx",
        "/EKAP/Ortak/YeniIhaleAramaData.ashx",
        "/EKAP/Teklif/IhaleDokumanDownload.aspx",
        "/EKAP/Istekli/IstekliBilgileri.ashx",
    ])
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (returns new dict)."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_settings(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load settings from YAML with optional overrides.

    Resolution order:
      1. Built-in defaults (Pydantic field defaults)
      2. ``configs/default.yaml``  (if exists)
      3. ``config_path``           (if provided)
      4. ``EKAP_CONFIG`` env-var   (if set, path to YAML)
      5. ``overrides`` dict        (if provided)
    """
    data: dict[str, Any] = {}

    # default.yaml
    default_path = Path("configs/default.yaml")
    if default_path.exists():
        with open(default_path) as fh:
            loaded = yaml.safe_load(fh) or {}
            data = _deep_merge(data, loaded)

    # explicit config path
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p) as fh:
                loaded = yaml.safe_load(fh) or {}
                data = _deep_merge(data, loaded)

    # env-var config
    env_cfg = os.environ.get("EKAP_CONFIG")
    if env_cfg:
        p = Path(env_cfg)
        if p.exists():
            with open(p) as fh:
                loaded = yaml.safe_load(fh) or {}
                data = _deep_merge(data, loaded)

    # programmatic overrides
    if overrides:
        data = _deep_merge(data, overrides)

    return Settings(**data)
