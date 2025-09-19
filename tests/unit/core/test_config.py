"""
Tests for configuration management in `core/config.py`.

Covers:
- Environment parsing and debug defaults
- Logging level coercion to the expected Literal
- API reload boolean parsing
- Allowed origins parsing
- get_model_config mapping
- get_config cache behavior
- AppConfig validation (debug only allowed in development)
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from core.config import (
    AIProviderConfig,
    APIConfig,
    AppConfig,
    DatabaseConfig,
    LoggingConfig,
    MonitoringConfig,
    get_config,
    get_model_config,
    load_config_from_env,
)


@pytest.fixture(autouse=True)
def clear_config_cache() -> Iterator[None]:
    """Ensure get_config cache is cleared before and after each test."""
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    yield
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def _set_minimal_valid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the minimal environment required for config to validate."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")


def test_load_config_dev_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_minimal_valid_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("API_RELOAD", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    config = load_config_from_env()

    assert config.environment == "development"
    assert config.debug is True
    assert config.api.reload is True  # defaults to debug
    assert config.logging.format == "console"
    assert config.logging.level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def test_api_reload_boolean_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_minimal_valid_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")

    # Explicit false should override debug default
    monkeypatch.setenv("API_RELOAD", "false")
    config = load_config_from_env()
    assert config.api.reload is False

    # Truthy values
    monkeypatch.setenv("API_RELOAD", "1")
    config = load_config_from_env()
    assert config.api.reload is True


def test_allowed_origins_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_minimal_valid_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("API_ALLOWED_ORIGINS", "http://a.example,http://b.example")

    config = load_config_from_env()

    assert config.api.allowed_origins == ["http://a.example", "http://b.example"]


def test_logging_level_literal_coercion(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_minimal_valid_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "staging")

    # Unknown level should coerce to INFO
    monkeypatch.setenv("LOG_LEVEL", "unknown")
    config = load_config_from_env()
    assert config.logging.level == "INFO"

    # Known level should pass through
    monkeypatch.setenv("LOG_LEVEL", "error")
    config = load_config_from_env()
    assert config.logging.level == "ERROR"


def test_get_model_config_maps_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_minimal_valid_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")
    # Override models via env to ensure mapping picks them up
    monkeypatch.setenv("ANOMALY_DETECTION_MODEL", "openai:gpt-4o-mini")
    monkeypatch.setenv("ROOT_CAUSE_MODEL", "openai:gpt-4o")

    cfg = load_config_from_env()

    anomaly_cfg = get_model_config("anomaly_detection")
    root_cfg = get_model_config("root_cause")

    assert anomaly_cfg["model_name"] == cfg.ai_provider.anomaly_detection_model
    assert root_cfg["model_name"] == cfg.ai_provider.root_cause_model
    # Shared defaults carried over
    for k in ("max_tokens", "temperature", "timeout_seconds", "max_retries"):
        assert anomaly_cfg[k] == getattr(cfg.ai_provider, f"default_{k}")
        assert root_cfg[k] == getattr(cfg.ai_provider, f"default_{k}")


def test_get_config_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_minimal_valid_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")

    # First call populates cache
    c1 = get_config()
    c2 = get_config()
    assert c1 is c2  # same object due to lru_cache


def test_app_config_debug_only_in_dev_validation() -> None:
    # Construct components with minimal content
    api = APIConfig()
    logging_cfg = LoggingConfig()
    ai = AIProviderConfig(
        openai_api_key="sk-test-openai",
        anthropic_api_key="sk-test-anthropic",
        anomaly_detection_model="openai:gpt-4o-mini",
    )
    monitoring = MonitoringConfig()
    database = DatabaseConfig()

    with pytest.raises(ValueError, match="debug mode is only allowed"):
        AppConfig(
            environment="production",
            debug=True,
            ai_provider=ai,
            monitoring=monitoring,
            database=database,
            api=api,
            logging=logging_cfg,
        )
