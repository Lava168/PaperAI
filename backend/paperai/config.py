from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _boolean(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    data_dir: Path
    database_path: Path
    model_base_url: str
    model_api_key: str
    model_name: str
    model_timeout: int
    max_source_chars: int
    allow_origins: tuple[str, ...]
    debug: bool

    @property
    def model_configured(self) -> bool:
        return bool(self.model_api_key and self.model_base_url and self.model_name)


def load_settings() -> Settings:
    data_dir = Path(os.getenv("PAPERAI_DATA_DIR", ROOT / "data")).expanduser().resolve()
    base_url = os.getenv(
        "PAPERAI_MODEL_BASE_URL",
        os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ).rstrip("/")
    api_key = os.getenv(
        "PAPERAI_MODEL_API_KEY",
        os.getenv("DASHSCOPE_API_KEY", os.getenv("QWEN_API_KEY", "")),
    )
    model = os.getenv("PAPERAI_MODEL", os.getenv("QWEN_MODEL", "qwen-plus"))
    origins = tuple(
        item.strip()
        for item in os.getenv("PAPERAI_ALLOW_ORIGINS", "http://127.0.0.1:8080,http://localhost:8080").split(",")
        if item.strip()
    )
    return Settings(
        host=os.getenv("PAPERAI_HOST", "127.0.0.1"),
        port=int(os.getenv("PAPERAI_PORT", os.getenv("PORT", "8080"))),
        data_dir=data_dir,
        database_path=Path(os.getenv("PAPERAI_DATABASE", data_dir / "paperai.db")).expanduser().resolve(),
        model_base_url=base_url,
        model_api_key=api_key,
        model_name=model,
        model_timeout=int(os.getenv("PAPERAI_MODEL_TIMEOUT", "180")),
        max_source_chars=int(os.getenv("PAPERAI_MAX_SOURCE_CHARS", "120000")),
        allow_origins=origins,
        debug=_boolean("PAPERAI_DEBUG"),
    )
