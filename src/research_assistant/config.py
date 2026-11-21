"""Application configuration: secrets from .env, everything else from config/settings.yaml."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_YAML = PROJECT_ROOT / "config" / "settings.yaml"
ENV_FILE = PROJECT_ROOT / ".env"

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class ModelsConfig(BaseModel):
    main: str
    judge: str
    fallback: str


class LLMConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class PathsConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")
    papers_file: Path = Path("data/papers.yaml")
    processed_dir: Path = Path("data/processed")
    strategies_file: Path = Path("config/strategies.yaml")
    cache_dir: Path = Path("data/cache")
    # JSONL audit log of every agent tool call (see research_assistant.security.tool_policy).
    audit_log: Path = Path("data/logs/tool_audit.jsonl")

    def resolve(self, root: Path = PROJECT_ROOT) -> PathsConfig:
        """Return a copy with relative paths anchored at the project root."""

        def _abs(p: Path) -> Path:
            return p if p.is_absolute() else root / p

        return PathsConfig(
            raw_dir=_abs(self.raw_dir),
            index_dir=_abs(self.index_dir),
            papers_file=_abs(self.papers_file),
            processed_dir=_abs(self.processed_dir),
            strategies_file=_abs(self.strategies_file),
            cache_dir=_abs(self.cache_dir),
            audit_log=_abs(self.audit_log),
        )


class ArxivConfig(BaseModel):
    api_url: str = "https://export.arxiv.org/api/query"
    eprint_url: str = "https://export.arxiv.org/e-print"
    pdf_url: str = "https://export.arxiv.org/pdf"
    # arXiv terms of use: at most one request every 3 seconds. Not configurable below that.
    min_interval_s: float = Field(default=3.0, ge=3.0)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    backoff_s: float = Field(default=3.0, ge=0)
    max_download_mb: int = Field(default=100, gt=0)
    max_extract_mb: int = Field(default=500, gt=0)
    user_agent: str = "research-assistant-corpus/0.1"


class SemanticScholarConfig(BaseModel):
    api_url: str = "https://api.semanticscholar.org/graph/v1"
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=4, ge=0)
    backoff_s: float = Field(default=2.0, ge=0)
    # Keyed access is limited to 1 request/s; unauthenticated traffic shares a global pool.
    min_interval_s: float = Field(default=1.0, ge=0)
    cache_ttl_h: float = Field(default=24.0, ge=0)
    user_agent: str = "research-assistant/0.1"


class LimitsConfig(BaseModel):
    max_question_chars: int = Field(default=2000, gt=0)


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=8, gt=0)
    bm25_weight: float = Field(default=0.5, ge=0.0)
    dense_weight: float = Field(default=0.5, ge=0.0)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Prepended to queries only (not documents). BGE v1.5 retrieval instruction; set to ""
    # for models that do not use one.
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = Field(default=32, gt=0)
    collection_name: str = "chunks"
    # Cross-encoder used by strategies with ``rerank: true``.
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_max_length: int = Field(default=256, gt=0)
    reranker_batch_size: int = Field(default=16, gt=0)
    # Reciprocal Rank Fusion constant: score = sum 1 / (rrf_k + rank).
    rrf_k: int = Field(default=60, gt=0)


class ChunkingConfig(BaseModel):
    """Chunk sizes are in tokens of ``encoding`` (tiktoken), used as an approximation."""

    max_tokens: int = Field(default=800, gt=0)
    overlap_tokens: int = Field(default=100, ge=0)
    encoding: str = "cl100k_base"

    @model_validator(mode="after")
    def _check_overlap(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class GenerationConfig(BaseModel):
    """Cited answer generation (see research_assistant.generation)."""

    # Spotlighting by datamarking (Hines et al. 2024): additionally replace every whitespace
    # run in source text with ``datamark_char``. Delimiters with a random boundary are always on.
    datamarking: bool = False
    datamark_char: str = Field(default=chr(0x02C6), min_length=1, max_length=1)
    # Length of the source excerpt stored on each Citation.
    excerpt_chars: int = Field(default=300, gt=0)


class AgentConfig(BaseModel):
    """LangGraph agent (see research_assistant.agent.graph)."""

    default_strategy: str = Field(default="hybrid", pattern=r"^[a-z0-9_]+$")
    # Completed earlier turns (question + final answer) passed to the model.
    history_turns: int = Field(default=6, ge=0, le=50)
    # Hard cap per turn; the policy never allows more than this many tool calls.
    max_tool_calls_per_turn: int = Field(default=6, ge=1, le=6)
    # Agent model calls per turn (each may request several tool calls).
    max_agent_steps: int = Field(default=6, ge=1, le=20)
    # Characters of one tool result shown to the model (JSON, spotlighted).
    max_tool_result_chars: int = Field(default=6000, gt=0)


class InjectionConfig(BaseModel):
    """Prompt-injection detection on user input."""

    warn_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    block_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    # Optional Hugging Face classifier (Apache-2.0, downloaded on first use).
    classifier_enabled: bool = False
    classifier_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    classifier_max_length: int = Field(default=512, gt=0)

    @model_validator(mode="after")
    def _check_thresholds(self) -> InjectionConfig:
        if self.warn_threshold > self.block_threshold:
            raise ValueError("warn_threshold must not exceed block_threshold")
        return self


class SecurityConfig(BaseModel):
    injection: InjectionConfig = InjectionConfig()
    # Single whitespace-free tokens longer than this are rejected (obfuscation, payloads).
    max_token_chars: int = Field(default=200, gt=0)


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Settings(BaseSettings):
    """Merged settings. Priority: init kwargs > environment > .env > settings.yaml."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        yaml_file=SETTINGS_YAML,
        env_nested_delimiter="__",
    )

    # Secrets (from environment / .env only).
    openrouter_api_key: SecretStr | None = None
    semantic_scholar_api_key: SecretStr | None = None

    # Non-secret config (from settings.yaml).
    models: ModelsConfig
    llm: LLMConfig = LLMConfig()
    paths: PathsConfig = PathsConfig()
    limits: LimitsConfig = LimitsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    arxiv: ArxivConfig = ArxivConfig()
    semantic_scholar: SemanticScholarConfig = SemanticScholarConfig()
    generation: GenerationConfig = GenerationConfig()
    agent: AgentConfig = AgentConfig()
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process and configure logging."""
    try:
        settings = Settings()  # type: ignore[call-arg]  # fields come from yaml/env sources
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {SETTINGS_YAML} or environment") from exc
    configure_logging(settings.logging.level)
    return settings


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once. Safe to call repeatedly."""
    root = logging.getLogger()
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        numeric = logging.INFO
    if not root.handlers:
        logging.basicConfig(level=numeric, format=_LOG_FORMAT)
    else:
        root.setLevel(numeric)
    # Keep noisy HTTP client logs (which may include URLs/headers) quiet.
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "pylatexenc", "pdfminer", "absl"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
