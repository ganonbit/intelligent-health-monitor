"""
Tests for AI analysis components.

Covers:
- AnomalyDetectionAgent: no metrics, low/high confidence handling
- RootCauseAnalysisAgent: successful analysis path
- IntelligentMonitoringService: orchestration and overall status determination

These tests avoid real API calls by patching the underlying Agent.run to return
pre-constructed results with a `.content` attribute.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from core.domain.models import AnomalyDetection, MetricType, Severity, SystemMetric
from core.services.ai_analysis import (
    AIAnalysisConfig,
    AnomalyDetectionAgent,
    IntelligentMonitoringService,
    RootCauseAnalysisAgent,
    SystemContext,
)


class _FakeAgentResult:
    """Minimal stand-in for pydantic-ai AgentRunResult with .content"""

    def __init__(self, content: Any) -> None:
        self.content = content


@pytest.fixture
def sample_metrics() -> list[SystemMetric]:
    now = datetime.now(UTC)
    return [
        SystemMetric(
            metric_type=MetricType.CPU,
            value=85.0,
            unit="percent",
            timestamp=now,
            source="web-1",
        ),
        SystemMetric(
            metric_type=MetricType.ERROR_RATE,
            value=10.0,
            unit="errors_per_minute",
            timestamp=now,
            source="web-1",
        ),
        SystemMetric(
            metric_type=MetricType.MEMORY,
            value=60.0,
            unit="percent",
            timestamp=now,
            source="web-1",
        ),
    ]


@pytest.fixture
def system_context() -> SystemContext:
    return SystemContext(
        system_name="web-api-cluster",
        system_type="web-server",
        environment="production",
        expected_load_pattern="peak-during-business-hours",
        recent_changes=["deploy v2.1.3"],
        baseline_metrics={"cpu": 45.0, "error_rate": 2.0, "memory": 60.0},
    )


@pytest.mark.asyncio
async def test_anomaly_detection_returns_none_on_no_metrics(system_context: SystemContext) -> None:
    agent = AnomalyDetectionAgent(AIAnalysisConfig())

    result = await agent.analyze_metrics([], system_context)

    assert result is None


@pytest.mark.asyncio
async def test_anomaly_detection_ignores_low_confidence(
    sample_metrics: list[SystemMetric], system_context: SystemContext
) -> None:
    config = AIAnalysisConfig(anomaly_threshold=0.7)
    agent = AnomalyDetectionAgent(config)

    low_conf_anomaly = AnomalyDetection(
        severity=Severity.LOW,
        confidence=0.5,  # below threshold
        affected_metrics=[MetricType.CPU],
        root_cause_hypothesis="Normal variance",
        recommended_actions=["Monitor"],
        correlation_window_minutes=10,
        model_reasoning="Test reasoning",
    )

    async def fake_run(*args, **kwargs):
        return _FakeAgentResult(low_conf_anomaly)

    # Patch the underlying pydantic-ai Agent.run
    agent.agent.run = fake_run  # type: ignore[assignment]

    result = await agent.analyze_metrics(sample_metrics, system_context)

    assert result is None


@pytest.mark.asyncio
async def test_anomaly_detection_returns_high_confidence_anomaly(
    sample_metrics: list[SystemMetric], system_context: SystemContext
) -> None:
    config = AIAnalysisConfig(anomaly_threshold=0.7)
    agent = AnomalyDetectionAgent(config)

    high_conf_anomaly = AnomalyDetection(
        severity=Severity.HIGH,
        confidence=0.92,  # above threshold
        affected_metrics=[MetricType.CPU, MetricType.ERROR_RATE],
        root_cause_hypothesis="CPU saturation correlates with error rate",
        recommended_actions=["Scale out", "Investigate deployment"],
        correlation_window_minutes=15,
        model_reasoning="Correlated spikes detected",
    )

    async def fake_run(*args, **kwargs):
        return _FakeAgentResult(high_conf_anomaly)

    agent.agent.run = fake_run  # type: ignore[assignment]

    result = await agent.analyze_metrics(sample_metrics, system_context)

    assert isinstance(result, AnomalyDetection)
    assert result.severity == Severity.HIGH
    assert result.confidence == pytest.approx(0.92, rel=1e-6)
    assert MetricType.CPU in result.affected_metrics


@pytest.mark.asyncio
async def test_root_cause_analysis_returns_text(system_context: SystemContext) -> None:
    rca = RootCauseAnalysisAgent(AIAnalysisConfig())

    async def fake_run(prompt: str, *args, **kwargs):
        assert "ANOMALY DETECTED" in prompt
        return _FakeAgentResult("ROOT CAUSE HYPOTHESIS: example")

    rca.agent.run = fake_run  # type: ignore[assignment]

    anomaly = AnomalyDetection(
        severity=Severity.MEDIUM,
        confidence=0.8,
        affected_metrics=[MetricType.CPU],
        root_cause_hypothesis="Hypothesis",
        recommended_actions=["Action"],
        correlation_window_minutes=10,
        model_reasoning="Reasoning",
    )

    text = await rca.analyze_root_cause(anomaly, [], system_context)
    assert isinstance(text, str)
    assert "ROOT CAUSE" in text


@pytest.mark.asyncio
async def test_intelligent_monitoring_service_orchestrates_and_sets_status(
    sample_metrics: list[SystemMetric], system_context: SystemContext
) -> None:
    service = IntelligentMonitoringService(AIAnalysisConfig())

    # Patch anomaly detection to return a high severity anomaly
    async def fake_detect(metrics, context):  # type: ignore[no-untyped-def]
        return AnomalyDetection(
            severity=Severity.HIGH,
            confidence=0.9,
            affected_metrics=[MetricType.CPU],
            root_cause_hypothesis="Hypothesis",
            recommended_actions=["Action"],
            correlation_window_minutes=10,
            model_reasoning="Reasoning",
        )

    service.anomaly_detection_agent.analyze_metrics = fake_detect  # type: ignore[assignment]

    # Patch RCA to avoid real LLM call
    async def fake_rca(anomaly, metrics, context):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0)  # ensure it's truly async
        return "analysis text"

    service.root_cause_analysis_agent.analyze_root_cause = fake_rca  # type: ignore[assignment]

    report = await service.analyze_system_health(sample_metrics, system_context)

    assert report.overall_status == "warning"  # HIGH -> warning
    assert len(report.anomalies) == 1
    assert report.analysis_duration_seconds > 0.0
