"""
Configuration management with environment variable support and validation.

Design principles:
- Environment-specific configs (dev, staging, prod)
- Validation at startup (fail fast)
- Type safety with Pydantic
- Secure defaults (no API keys in code)
"""

import os
from functools import lru_cache
from typing import Any, Literal, cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

# Load environment variables from .env file
load_dotenv()


class AIProviderConfig(BaseModel):
    """AI provider configuration with secure defaults."""

    openai_api_key: str = Field(..., description="OpenAI API key")
    anthropic_api_key: str | None = Field(None, description="Anthropic API key (optional)")

    # Model selection for different tasks
    anomaly_detection_model: str = Field(
        default="openai:gpt-4o-mini", description="Model to use for anomaly detection"
    )
    root_cause_model: str = Field(
        default="openai:gpt-4o", description="Model to use for root cause analysis"
    )

    # AI behavior settings
    default_temperature: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Default temperature for AI models"
    )
    default_max_tokens: int = Field(
        default=1000, gt=0, description="Default max tokens for AI models"
    )
    default_timeout_seconds: float = Field(
        default=30.0, gt=0.0, description="Default timeout for AI models"
    )
    default_max_retries: int = Field(
        default=3, ge=0, description="Default max retries for AI models"
    )

    @field_validator("openai_api_key", "anthropic_api_key")
    def validate_api_keys(cls, v):
        if not v or v == "your-openai-api-key-here":
            raise ValueError("AI provider API key must be set in environment or .env file")
        if not v.startswith("sk-"):
            raise ValueError("AI provider API key must start with 'sk-'")
        return v


class MonitoringConfig(BaseModel):
    """Core monitoring system configuration."""

    collection_interval_seconds: float = Field(
        default=30.0, gt=0.0, description="Interval between metric collections"
    )
    analysis_interval_seconds: float = Field(
        default=60.0, gt=0.0, description="Interval between AI analysis"
    )
    retention_days: int = Field(default=30, gt=0, description="Number of days to retain metrics")

    # Performance tuning
    max_concurrent_sources: int = Field(
        default=10, gt=0, description="Maximum number of concurrent metric sources"
    )
    max_concurrent_analyses: int = Field(
        default=5, gt=0, description="Maximum number of concurrent AI analyses"
    )
    collection_timeout_seconds: float = Field(
        default=10.0, gt=0.0, description="Timeout for metric collection"
    )

    # Alert settings
    anomaly_confidence_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Threshold for anomaly confidence"
    )
    enable_predictive_analysis: bool = Field(default=True, description="Enable predictive analysis")
    correlation_window_minutes: int = Field(
        default=30, gt=0, description="Window for correlation analysis"
    )


class DatabaseConfig(BaseModel):
    """Database configuration (for future use)."""

    url: str = Field(default="sqlite:///./monitoring.db", description="Database URL")
    pool_size: int = Field(default=10, gt=0, description="Database connection pool size")
    max_overflow: int = Field(default=20, ge=0, description="Database connection pool overflow")
    pool_timeout: int = Field(default=30, gt=0, description="Database connection pool timeout")


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, gt=0, lt=65536, description="API server port")
    reload: bool = Field(default=False, description="Enable auto-reload for development")

    # Security settings
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"], description="Allowed origins for CORS"
    )
    api_key_header: str = Field(default="X-API-Key", description="Header name for API key")

    # Performance settings
    worker_count: int = Field(default=1, gt=0, description="Number of worker processes")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    format: Literal["json", "console"] = Field(default="json", description="Logging format")
    enable_metrics: bool = Field(default=True, description="Enable metrics logging")

    # Log destinations
    enable_file_logging: bool = Field(default=False, description="Enable file logging")
    log_file_path: str = Field(default="./logs/monitoring.log", description="Path to log file")


class AppConfig(BaseModel):
    """Main application configuration combining all subsystems."""

    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # Component configs
    ai_provider: AIProviderConfig
    monitoring: MonitoringConfig
    database: DatabaseConfig
    api: APIConfig
    logging: LoggingConfig

    @model_validator(mode="after")
    def debug_only_in_dev(self) -> "AppConfig":
        """Ensure debug mode is only allowed in development environment."""
        if self.debug and self.environment != "development":
            raise ValueError("debug mode is only allowed in development environment")
        return self

    class Config:
        env_nested_delimiter = "__"  # Allows MONITORING__COLLECTION_INTERVAL_SECONDS


def load_config_from_env() -> AppConfig:
    """Load configuration from environment variables with validation."""

    def _env_to_literal(val: str) -> Literal["development", "staging", "production"]:
        v = val.strip().lower()
        if v in {"dev", "development"}:
            return "development"
        if v in {"stage", "staging"}:
            return "staging"
        return "production"

    def _level_to_literal(val: str) -> Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        v = val.strip().upper()
        return cast(
            Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            v if v in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO",
        )

    def _parse_bool(val: str | None, default: bool) -> bool:
        if val is None:
            return default
        return val.strip().lower() in {"1", "true", "yes", "on"}

    # Detect environment
    environment = _env_to_literal(os.getenv("ENVIRONMENT", "development"))
    debug = environment == "development"

    # AI Provider config from environment
    ai_config = AIProviderConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anomaly_detection_model=os.getenv("ANOMALY_DETECTION_MODEL", "openai:gpt-4o-mini"),
        root_cause_model=os.getenv("ROOT_CAUSE_MODEL", "openai:gpt-4o"),
    )

    # Monitoring config with environment overrides
    monitoring_config = MonitoringConfig(
        collection_interval_seconds=float(os.getenv("COLLECTION_INTERVAL_SECONDS", "30.0")),
        analysis_interval_seconds=float(os.getenv("ANALYSIS_INTERVAL_SECONDS", "60.0")),
        anomaly_confidence_threshold=float(os.getenv("ANOMALY_CONFIDENCE_THRESHOLD", "0.75")),
    )

    # Database config from environment
    database_config = DatabaseConfig(
        url=os.getenv("DATABASE_URL", "sqlite:///./monitoring.db"),
    )

    # API config
    api_config = APIConfig(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=_parse_bool(os.getenv("API_RELOAD"), debug),
        allowed_origins=os.getenv("API_ALLOWED_ORIGINS", "http://localhost:3000").split(","),
        worker_count=int(os.getenv("API_WORKER_COUNT", "1")),
    )

    # Logging config
    logging_config = LoggingConfig(
        level=_level_to_literal(os.getenv("LOG_LEVEL", "INFO")),
        format="console" if debug else "json",
    )

    # Application config
    return AppConfig(
        environment=environment,
        debug=debug,
        ai_provider=ai_config,
        monitoring=monitoring_config,
        database=database_config,
        api=api_config,
        logging=logging_config,
    )


@lru_cache
def get_config() -> AppConfig:
    """Get cached application configuration."""
    return load_config_from_env()


# Configuration validation and helpers
def validate_config() -> None:
    """Validate configuration at startup."""
    try:
        config = get_config()
        print(f"✅ Configuration loaded for {config.environment} environment")

        # Test AI provider connection (optional)
        if config.ai_provider.openai_api_key:
            print("✅ OpenAI API key configured")

        if config.ai_provider.anthropic_api_key:
            print("✅ Anthropic API key configured")

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        raise


def get_model_config(task: Literal["anomaly_detection", "root_cause"]) -> dict[str, Any]:
    """Get model configuration based on task."""
    config = get_config()

    if task == "anomaly_detection":
        return {
            "model_name": config.ai_provider.anomaly_detection_model,
            "max_tokens": config.ai_provider.default_max_tokens,
            "temperature": config.ai_provider.default_temperature,
            "timeout_seconds": config.ai_provider.default_timeout_seconds,
            "max_retries": config.ai_provider.default_max_retries,
        }
    elif task == "root_cause":
        return {
            "model_name": config.ai_provider.root_cause_model,
            "max_tokens": config.ai_provider.default_max_tokens,
            "temperature": config.ai_provider.default_temperature,
            "timeout_seconds": config.ai_provider.default_timeout_seconds,
            "max_retries": config.ai_provider.default_max_retries,
        }
    else:
        raise ValueError(f"Unknown task: {task}")


# Development helpers
def print_config_summary() -> None:
    """Print configuration summary for debugging."""
    config = get_config()

    print("\n🔧 CONFIGURATION SUMMARY")
    print(f"Environment: {config.environment}")
    print(f"Debug Mode: {config.debug}")
    print(f"Log Level: {config.logging.level}")

    print("\n🤖 AI CONFIGURATION")
    print(f"Anomaly Model: {config.ai_provider.anomaly_detection_model}")
    print(f"Root Cause Model: {config.ai_provider.root_cause_model}")
    print(f"Confidence Threshold: {config.monitoring.anomaly_confidence_threshold:.1%}")

    print("\n📊 MONITORING CONFIGURATION")
    print(f"Collection Interval: {config.monitoring.collection_interval_seconds}s")
    print(f"Analysis Interval: {config.monitoring.analysis_interval_seconds}s")
    print(f"Correlation Window: {config.monitoring.correlation_window_minutes}m")

    print("\n🌐 API CONFIGURATION")
    print(f"Host: {config.api.host}:{config.api.port}")
    print(f"Reload: {config.api.reload}")


if __name__ == "__main__":
    # Test configuration loading
    validate_config()
    print_config_summary()
