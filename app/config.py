from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

_ENV_FILE_LOADED = load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    ENV_FILE_LOADED: bool = _ENV_FILE_LOADED
    RUNTIME_MODE: str = os.getenv("RUNTIME_MODE", "local").strip().lower()
    DATA_SOURCE: str = os.getenv("DATA_SOURCE", "repo").strip().lower()

    GCP_PROJECT: str | None = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or None
    BQ_DATASET: str | None = os.getenv("BQ_DATASET") or None
    REGION: str = os.getenv("REGION", "us-central1")
    GCS_BUCKET: str | None = os.getenv("GCS_BUCKET") or None

    USE_BIGQUERY: bool = _env_bool("USE_BIGQUERY", default=False)
    ENABLE_GEMINI: bool = _env_bool("ENABLE_GEMINI", default=False)

    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
    GEMINI_PROVIDER: str = os.getenv("GEMINI_PROVIDER", "api_key").strip().lower()
    GEMINI_LOCATION: str = os.getenv("GEMINI_LOCATION", os.getenv("REGION", "us-central1"))
    GEMINI_MODEL_FAST: str = os.getenv("GEMINI_MODEL_FAST", "gemini-2.0-flash")
    GEMINI_MODEL_REASONING: str = os.getenv("GEMINI_MODEL_REASONING", "gemini-2.5-pro")
    CHAT_PROVIDER: str = os.getenv("CHAT_PROVIDER", "offline").strip().lower()
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))
    OLLAMA_NUM_GPU: int | None = int(os.getenv("OLLAMA_NUM_GPU")) if os.getenv("OLLAMA_NUM_GPU") not in {None, ""} else None
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
    OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "220"))
    OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
    OLLAMA_FAST_KNOWN_ANSWERS: bool = _env_bool("OLLAMA_FAST_KNOWN_ANSWERS", default=True)

    POLICY_ARTIFACT_PATH: str = os.getenv("POLICY_ARTIFACT_PATH", "artifacts/q_policy.json")
    POLICY_METRICS_PATH: str = os.getenv("POLICY_METRICS_PATH", "artifacts/policy_metrics.json")
    LOCAL_LOG_PATH: str = os.getenv("LOCAL_LOG_PATH", "artifacts/local_logs.jsonl")

    def __post_init__(self) -> None:
        # Keep old values compatible, but expose only two honest runtime paths:
        # local = bundled CSV/artifacts/JSONL/deterministic fallback
        # cloud = BigQuery/Gemini where credentials and env flags are configured
        if self.RUNTIME_MODE in {"demo", "offline"}:
            self.RUNTIME_MODE = "local"
        if self.RUNTIME_MODE == "auto":
            self.RUNTIME_MODE = "cloud"
        if self.RUNTIME_MODE not in {"local", "cloud"}:
            self.RUNTIME_MODE = "local"
        if self.DATA_SOURCE not in {"repo", "bigquery"}:
            self.DATA_SOURCE = "repo"
        if self.GEMINI_PROVIDER not in {"api_key", "vertex"}:
            self.GEMINI_PROVIDER = "api_key"

    @property
    def is_local_mode(self) -> bool:
        return self.RUNTIME_MODE == "local"

    @property
    def is_demo_mode(self) -> bool:
        # Backward-compatible alias used by older code.
        return self.is_local_mode

    @property
    def bigquery_configured(self) -> bool:
        return bool(self.GCP_PROJECT and self.BQ_DATASET)

    @property
    def gemini_configured(self) -> bool:
        if self.GEMINI_PROVIDER == "vertex":
            return bool(self.GCP_PROJECT and self.GEMINI_LOCATION)
        return bool(self.GEMINI_API_KEY)

    @property
    def use_bigquery(self) -> bool:
        if self.is_demo_mode:
            return False
        return self.DATA_SOURCE == "bigquery" and self.USE_BIGQUERY and self.bigquery_configured

    @property
    def use_gemini(self) -> bool:
        if self.is_demo_mode:
            return False
        return self.ENABLE_GEMINI and self.gemini_configured

    def cloud_available(self) -> bool:
        return bool((self.USE_BIGQUERY and self.bigquery_configured) or (self.ENABLE_GEMINI and self.gemini_configured))

    def set_runtime_mode(self, mode: str) -> str:
        normalized = (mode or "local").strip().lower()
        if normalized in {"demo", "offline"}:
            normalized = "local"
        if normalized == "auto":
            normalized = "cloud"
        if normalized not in {"local", "cloud"}:
            raise ValueError("runtime mode must be local or cloud")
        if normalized == "cloud" and not self.cloud_available():
            # Stay honest: do not let the UI imply cloud mode when credentials/env flags are absent.
            self.RUNTIME_MODE = "local"
            self.DATA_SOURCE = "repo"
            return self.RUNTIME_MODE
        self.RUNTIME_MODE = normalized
        if normalized == "local":
            self.DATA_SOURCE = "repo"
        elif normalized == "cloud" and self.USE_BIGQUERY and self.bigquery_configured:
            self.DATA_SOURCE = "bigquery"
        return self.RUNTIME_MODE


settings = Settings()
