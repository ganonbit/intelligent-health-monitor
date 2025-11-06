"""
Domain models for system health monitoring.

These models represent the core business concepts and are framework-agnostic.
They use Pydantic for validation but could be swapped to dataclasses if needed.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MetricType(str, Enum):
    """Types of system metrics we can collect."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK = "network"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    CUSTOM = "custom"


class Severity(str, Enum):
    """Alert severity levels following standard SRE practices."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemMetric(BaseModel):
    """Individual system metric reading."""

    model_config = ConfigDict(frozen=True)  # Immutable for better reasoning

    metric_type: MetricType = Field(default=MetricType.CUSTOM)
    name: str | None = Field(default=None, description="Name of the metric")
    value: float
    unit: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = Field(description="Source system/service name")
    tags: dict[str, str] = Field(default_factory=dict)


class AnomalyDetection(BaseModel):
    """
    AI analysis result for potential system anomalies.

    Implements the "Verbal Confidence" pattern where the LLM self-assesses its certainty
    and populates the confidence_score field based on data quality, pattern clarity,
    and analysis completeness. This approach is provider-agnostic and works across
    all LLM providers (OpenAI, Anthropic, Google, etc.).
    """

    severity: Severity
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Self-assessed confidence level (0.0-1.0). The LLM evaluates its certainty "
            "by considering: data quality, pattern clarity, correlation strength, "
            "historical context availability, and analysis completeness. "
            "Use 0.9-1.0 for very high confidence, 0.7-0.9 for high confidence, "
            "0.5-0.7 for moderate confidence, and below 0.5 for low confidence."
        ),
    )
    affected_metrics: list[MetricType]
    root_cause_hypothesis: str = Field(min_length=10, max_length=500)
    recommended_actions: list[str] = Field(min_length=1, max_length=5)
    correlation_window_minutes: float = Field(
        gt=0.0, description="Time window analyzed for correlation"
    )
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Metadata for debugging AI decisions
    model_reasoning: str = Field(
        description=(
            "Internal AI step-by-step reasoning process. Explain your analysis methodology, "
            "what patterns you identified, why you assigned this confidence level, "
            "and any uncertainties or limitations in the available data."
        )
    )
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HealthReport(BaseModel):
    """Comprehensive health report for a system."""

    overall_status: Literal["healthy", "warning", "critical"]
    metrics: list[SystemMetric]
    anomalies: list[AnomalyDetection]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    analysis_duration_seconds: float = Field(gt=0.0)
