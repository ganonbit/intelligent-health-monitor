"""
AI-powered system analysis using Pydantic AI.

Key architectural decisions:
- Multi-agent system: Different agents for different analysis types
- Type-safe AI responses: All AI output validated with Pydantic
- Context-aware analysis: Agents understand system topology and history
- Fallback strategies: Graceful degradation when AI services fail

Why Pydantic AI over alternatives:
- Type safety: AI responses are validated, not just strings
- Composability: Agents can call other agents for complex analysis
- Debugging: Full visibility into AI decision-making process
- Production-ready: Built-in retries, timeouts, error handling
"""

import asyncio
import textwrap
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from core.domain.models import AnomalyDetection, HealthReport, MetricType, Severity, SystemMetric

logger = structlog.get_logger(__name__)

# Overall status type alias to keep annotations short and within line length limits
OverallStatus = Literal["healthy", "warning", "critical"]


# Enhanced domain models for AI analysis
class MetricCorrelation(BaseModel):
    """Represents correlation between different metrics."""

    primary_metric: MetricType
    correlated_metrics: list[MetricType]
    correlation_strength: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    time_window_minutes: int = Field(gt=0)


class SystemContext(BaseModel):
    """Context about the system being monitored for AI analysis."""

    system_name: str
    system_type: str = Field(description="e.g., web-server, database, hvac-chiller")
    environment: str = Field(default="production")
    expected_load_pattern: str = Field(description="e.g., business-hours, 24x7, batch-processing")
    recent_changes: list[str] = Field(
        default_factory=list, description="Recent deployments, configuration changes, etc."
    )
    baseline_metrics: dict[str, float] = Field(
        default_factory=dict, description="Normal operating ranges for key metrics"
    )


class AIAnalysisConfig(BaseModel):
    """Configuration for AI analysis with smart defaults."""

    # Use explicit string to satisfy Agent model literal type requirements
    model_name: str = "openai:gpt-4o-mini"
    max_tokens: int = Field(default=1000, gt=100)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)  # Low for consistent analysis
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)

    # Analysis-specific configuration
    correlation_window_minutes: int = Field(default=30, gt=0)
    anomaly_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    required_multiple_metrics: bool = Field(
        default=True, description="Require multiple metrics to confirm anomalies"
    )


class AnomalyDetectionAgent(Agent[AnomalyDetection]):
    """
    AI agent specialized in detecting system anomalies.

    Design principles:
    - Single responsibility: Only detects anomalies, doesn't recommend actions
    - Context-aware: Uses system context and metric history
    - Confident: Only returns high-confidence detections
    - Explainable: Always provides reasoning for decisions
    """

    def __init__(self, config: AIAnalysisConfig) -> None:
        self.config = config
        self.logger = logger.bind(component="anomaly_detection_agent")

        # Initialize the agent with structured output
        self.agent = Agent(
            model=self.config.model_name,
            output_type=AnomalyDetection,
            system_prompt=self._build_system_prompt(),
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt that creates an expert SRE personality."""
        return f"""You are a Senior Site Reliability Engineer with 15 years of experience
analyzing system metrics and identifying anomalies before they cause outages.

Your job is to analyze system metrics and identify potential problems with high confidence.

Key principles:
1. ONLY flag anomalies you're confident about (>{self.config.anomaly_threshold * 100}% confidence)
2. Look for CORRELATIONS between metrics, not just individual spikes
3. Consider CONTEXT: time of day, system type, recent changes
4. Provide ACTIONABLE insights, not just "CPU is high"
5. Use your experience to distinguish normal variance from real problems

Analysis window: {self.config.correlation_window_minutes} minutes
Severity levels:
- LOW: Minor deviation, monitor closely
- MEDIUM: Significant deviation, investigate soon
- HIGH: Major deviation, investigate immediately
- CRITICAL: System likely failing, immediate action required

Always explain your reasoning step-by-step so other engineers can understand your analysis."""

    async def analyze_metrics(
        self, metrics: list[SystemMetric], context: SystemContext
    ) -> AnomalyDetection | None:
        """
        Analyze metrics for anomalies with full context.

        Returns None if no significant anomalies detected.
        """
        if not metrics:
            self.logger.warning("no_metrics_provided", system_name=context.system_name)
            return None

        start_time = datetime.now(UTC)

        try:
            # Prepare context for AI analysis
            analysis_context = self._prepare_analysis_context(metrics, context)
            self.logger.debug("analysis_context_prepared", analysis_context=analysis_context)

            # Run AI analysis with timeout and error handling
            result = await asyncio.wait_for(
                self.agent.run(
                    user_prompt=self._build_user_prompt(metrics, context, analysis_context),
                    message_history=[],  # Could add conversation history for multi-turn
                ),
                timeout=self.config.timeout_seconds,
            )

            # Validate AI response meets our confidence threshold
            anomaly = cast(AnomalyDetection, cast(Any, result).content)
            if anomaly.confidence < self.config.anomaly_threshold:
                self.logger.info(
                    "low_confidence_anomaly_ignored",
                    confidence=anomaly.confidence,
                    threshold=self.config.anomaly_threshold,
                )
                return None

            analysis_duration = (datetime.now(UTC) - start_time).total_seconds()

            self.logger.info(
                "anomaly_detected",
                severity=anomaly.severity,
                confidence=anomaly.confidence,
                affected_metrics=anomaly.affected_metrics,
                duration_seconds=round(analysis_duration, 3),
            )

            return anomaly

        except TimeoutError:
            self.logger.error("ai_analysis_timeout", timeout_seconds=self.config.timeout_seconds)
            return None
        except Exception as e:
            self.logger.error("ai_analysis_failed", error=str(e))
            return None

    def _prepare_analysis_context(
        self,
        metrics: list[SystemMetric],
        context: SystemContext,
    ) -> dict[str, Any]:
        """Prepare context for AI analysis."""

        # Group metrics by type for pattern analysis
        metrics_by_type: defaultdict[MetricType, list[float]] = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type].append(metric.value)

        # Calculate basic statistics for each metric type
        metric_stats = {}
        for metric_type, values in metrics_by_type.items():
            if values:
                metric_stats[metric_type.value] = {
                    "current": values[-1] if values else 0,
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "count": len(values),
                }

        # Time span analysis
        if metrics:
            time_span_minutes = (
                max(m.timestamp for m in metrics) - min(m.timestamp for m in metrics)
            ).total_seconds() / 60
        else:
            time_span_minutes = 0

        return {
            "metrics_statistics": metric_stats,
            "time_span_minutes": round(time_span_minutes, 1),
            "total_data_points": len(metrics),
            "unique_sources": len({m.source for m in metrics}),
            "system_context": context.model_dump(),
        }

    def _build_user_prompt(
        self,
        metrics: list[SystemMetric],
        context: SystemContext,
        analysis_context: dict[str, Any],
    ) -> str:
        """Build specific analysis prompt for current metrics."""

        # Recent metrics summary
        recent_metrics = sorted(metrics, key=lambda m: m.timestamp, reverse=True)[:10]
        metrics_summary = []

        for metric in recent_metrics:
            metrics_summary.append(
                f"{metric.metric_type.value}: {metric.value} {metric.unit} "
                f"from {metric.source} at {metric.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        # System context
        context_info = []
        if context.recent_changes:
            context_info.append(f"Recent changes: {', '.join(context.recent_changes)}")

        if context.baseline_metrics:
            baseline_info = []
            for metric_name, baseline in context.baseline_metrics.items():
                baseline_info.append(f"{metric_name}: {baseline}")
            context_info.append(f"Baseline metrics: {', '.join(baseline_info)}")

        # Analysis context summary
        metrics_stats = analysis_context.get("metrics_statistics", {})
        metrics_stats_lines = []
        for name, v in metrics_stats.items():
            line = (
                f"- {name}: current={v['current']}, min={v['min']}, max={v['max']},"
                f" avg={v['avg']:.1f}, count={v['count']}"
            )
            metrics_stats_lines.append(textwrap.fill(line, width=100))

        return f"""Analyze these system metrics for anomalies:
SYSTEM: {context.system_name} ({context.system_type})
ENVIRONMENT: {context.environment}
LOAD PATTERN: {context.expected_load_pattern}

RECENT METRICS ({len(recent_metrics)} most recent):
{chr(10).join(metrics_summary)}

CONTEXT:
{chr(10).join(context_info) if context_info else "No additional context"}

ANALYSIS CONTEXT SUMMARY:
TIME SPAN (minutes): {analysis_context.get("time_span_minutes", "n/a")}
TOTAL DATA POINTS: {analysis_context.get("total_data_points", "n/a")}
UNIQUE SOURCES: {analysis_context.get("unique_sources", "n/a")}

METRICS STATISTICS:
{chr(10).join(metrics_stats_lines) if metrics_stats_lines else "No statistics available"}

TIME WINDOW: {self.config.correlation_window_minutes} minutes

Please analyze for anomalies considering:
1. Are any metrics significantly outside normal ranges?
2. Do you see concerning correlations between metrics?
3. Given the system type and context, is this behavior problematic?
4. What might be the root cause if there is an issue?

Only report anomalies you're confident about (>{self.config.anomaly_threshold * 100}% sure)."""


class RootCauseAnalysisAgent:
    """
    AI agent specialized in root cause analysis and recommendations.

    Takes anomaly detections and provides deeper analysis and actionable recommendations.
    This separation allows for different models/prompts optimized for different tasks.
    """

    def __init__(self, config: AIAnalysisConfig) -> None:
        self.config = config
        self.logger = logger.bind(component="root_cause_analysis_agent")

        # Initialize the agent with structured output
        self.agent = Agent(
            model=self.config.model_name,
            output_type=str,  # Free form analysis output
            system_prompt=self._build_system_prompt(),
        )

    def _build_system_prompt(self) -> str:
        return """You are a Senior Systems Architect specializing in root cause analysis.

Given an anomaly detection, provide deep analysis of potential root causes and
specific, actionable recommendations.

Structure your response as:
1. ROOT CAUSE HYPOTHESIS: Most likely explanation
2. SUPPORTING EVIDENCE: Why this explanation fits the data
3. ALTERNATIVE CAUSES: Other possibilities to investigate
4. IMMEDIATE ACTIONS: What to do right now
5. PREVENTIVE MEASURES: How to avoid this in the future

Be specific and actionable. Don't just say "check the logs" -
say "check application logs for OutOfMemoryError exceptions in the last 30 minutes"."""

    async def analyze_root_cause(
        self,
        anomaly: AnomalyDetection,
        metrics: list[SystemMetric],
        context: SystemContext,
    ) -> str:
        """Perform detailed root cause analysis."""

        prompt = f"""ANOMALY DETECTED:
Severity: {anomaly.severity}
Confidence: {anomaly.confidence:.2%}
Affected Metrics: {", ".join(m.value for m in anomaly.affected_metrics)}
Initial Hypothesis: {anomaly.root_cause_hypothesis}

SYSTEM CONTEXT:
System: {context.system_name} ({context.system_type})
Environment: {context.environment}
Recent Changes: {", ".join(context.recent_changes) or "None"}

METRIC TRENDS:
{self._format_metric_trends(metrics, anomaly.affected_metrics)}

Please provide detailed root cause analysis and specific recommendations."""

        try:
            result = await self.agent.run(prompt)
            return cast(str, cast(Any, result).content)
        except Exception as e:
            self.logger.error("root_cause_analysis_failed", error=str(e))
            return f"Root cause analysis failed: {str(e)}"

    def _format_metric_trends(
        self,
        metrics: list[SystemMetric],
        affected_metrics: list[MetricType],
    ) -> str:
        """Format metric trends for the root cause analysis prompt."""
        trends = []
        for metric_type in affected_metrics:
            relevant_metrics = [m for m in metrics if m.metric_type == metric_type]
            if not relevant_metrics:
                continue

            # sort by timestamp
            relevant_metrics.sort(key=lambda m: m.timestamp)

            # show trend over time
            if len(relevant_metrics) >= 3:
                recent_avg = sum(m.value for m in relevant_metrics[-3:]) / 3
                earlier_avg = sum(m.value for m in relevant_metrics[:3]) / 3
                trend_direction = "up ↑" if recent_avg > earlier_avg else "down ↓"
                trend_change = abs(recent_avg - earlier_avg)
                unit = relevant_metrics[-1].unit

                trends.append(
                    f"{metric_type.value}: {trend_direction}"
                    f"{earlier_avg:.1f} -> {recent_avg:.1f}"
                    f"({trend_change:.1f} {unit})"
                )

        return "\n".join(trends) if trends else "Insufficient data for trend analysis"


class IntelligentMonitoringService:
    """
    Orchestrates multiple AI agents for comprehensive system analysis.

    This is the main service that coordinates anomaly detection and root cause analysis,
    handles fallbacks, and maintains analysis history.
    """

    def __init__(self, config: AIAnalysisConfig) -> None:
        self.config = config
        self.logger = logger.bind(component="intelligent_monitoring_service")

        # Initialize AI agents
        self.anomaly_detection_agent = AnomalyDetectionAgent(config)
        self.root_cause_analysis_agent = RootCauseAnalysisAgent(config)

        # Simple in-memory history for context (in production: use proper storage)
        self.analysis_history: list[tuple[datetime, AnomalyDetection]] = []

    async def analyze_system_health(
        self,
        metrics: list[SystemMetric],
        context: SystemContext,
    ) -> HealthReport:
        """
        Comprehensive system health analysis using multiple AI agents.

        Returns a complete health report with AI-powered insights.
        """

        analysis_start = datetime.now(UTC)
        self.logger.info(
            "system_health_analysis_started", start_time=analysis_start, metrics_count=len(metrics)
        )

        try:
            # Step 1: Detect anomalies
            anomaly = await self.anomaly_detection_agent.analyze_metrics(metrics, context)
            anomalies = [anomaly] if anomaly else []

            # Step 2: Analyze root cause
            if anomaly:
                detailed_analysis = await self.root_cause_analysis_agent.analyze_root_cause(
                    anomaly, metrics, context
                )

                # Not attaching to anomaly to satisfy typing (no such field on AnomalyDetection)
                self.logger.debug("root_cause_analysis_completed", preview=detailed_analysis[:200])

                # Store in history for future context
                self.analysis_history.append((datetime.now(UTC), anomaly))

                # Keep history manageable (last 100 analyses)
                if len(self.analysis_history) > 100:
                    self.analysis_history = self.analysis_history[-100:]

            # Step 3: Determine overall system status
            overall_status = self._determine_overall_status(anomalies)

            analysis_duration = (datetime.now(UTC) - analysis_start).total_seconds()

            health_report = HealthReport(
                overall_status=overall_status,
                metrics=metrics,
                anomalies=anomalies,
                analysis_duration_seconds=analysis_duration,
            )

            self.logger.info(
                "system_health_analysis_completed",
                duration_seconds=round(analysis_duration, 3),
                overall_status=overall_status,
                anomalies_detected=len(anomalies),
            )

            return health_report

        except Exception as e:
            self.logger.error("system_health_analysis_failed", error=str(e))

            # Fallback: Create basic health report without AI analysis
            return HealthReport(
                overall_status="warning",  # Conservative fallback
                metrics=metrics,
                anomalies=[],
                analysis_duration_seconds=(datetime.now(UTC) - analysis_start).total_seconds(),
            )

    def _determine_overall_status(self, anomalies: list[AnomalyDetection]) -> OverallStatus:
        """Determine overall system status based on detected anomalies."""
        if not anomalies:
            return "healthy"

        # Check for critical anomalies
        if any(a.severity == Severity.CRITICAL for a in anomalies):
            return "critical"

        # Check for high severity anomalies
        if any(a.severity == Severity.HIGH for a in anomalies):
            return "warning"

        # Check for medium or low severity anomalies
        return "warning"

    async def get_analysis_history(
        self, hours: int = 24
    ) -> list[tuple[datetime, AnomalyDetection]]:
        """Get recent analysis history for trending and context."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        return [
            (timestamp, anomaly)
            for timestamp, anomaly in self.analysis_history
            if timestamp >= cutoff
        ]


async def main() -> None:
    """Demonstrate AI analysis of system health."""

    # Configuration
    config = AIAnalysisConfig(
        model_name="openai:gpt-4o-mini",  # Cost effective for development
        temperature=0.1,  # Conservative for critical analysis
        anomaly_threshold=0.75,  # High confidence for critical analysis
    )

    # System context (in production this comes from configuration/discovery)
    system_context = SystemContext(
        system_name="web-api-cluster",
        system_type="web-server",
        environment="production",
        expected_load_pattern="peak-during-business-hours",
        recent_changes=["deployed v2.1.3", "increased replica count to 5"],
        baseline_metrics={
            "cpu": 45.0,
            "memory": 60.0,
            "error_rate": 2.0,
        },
    )

    # Simulate problematic metrics (high cpu + high error rate correlation)
    from core.domain.models import MetricType, SystemMetric

    now = datetime.now(UTC)
    problematic_metrics = [
        SystemMetric(
            metric_type=MetricType.CPU,
            value=85.5,  # Well above baseline
            unit="percent",
            timestamp=now,
            source="web-server-1",
        ),
        SystemMetric(
            metric_type=MetricType.ERROR_RATE,
            value=12.3,  # Much higher than baseline
            unit="errors_per_minute",
            timestamp=now,
            source="web-server-1",
        ),
        SystemMetric(
            metric_type=MetricType.MEMORY,
            value=58.2,  # Normal
            unit="percent",
            timestamp=now,
            source="web-server-1",
        ),
    ]

    # Initialize monitoring service
    monitoring_service = IntelligentMonitoringService(config)

    # Analyze system health
    health_report = await monitoring_service.analyze_system_health(
        problematic_metrics, system_context
    )

    # Display results
    print("\n🏥 SYSTEM HEALTH REPORT")
    print(f"Overall Status: {health_report.overall_status.upper()}")
    print(f"Analysis Duration: {health_report.analysis_duration_seconds:.2f}s")
    print(f"Metrics Analyzed: {len(health_report.metrics)}")

    if health_report.anomalies:
        print(f"Anomalies Detected: {len(health_report.anomalies)}")
        print("\nANOMALIES:")
        for i, anomaly in enumerate(health_report.anomalies, 1):
            print(f"\n🚨 ANOMALY #{i}")
            print(f"Severity: {anomaly.severity}")
            print(f"Confidence: {anomaly.confidence:.1%}")
            print(f"Affected Metrics: {', '.join(m.value for m in anomaly.affected_metrics)}")
            print(f"Root Cause Hypothesis: {anomaly.root_cause_hypothesis}")
            print(f"Recommended Actions: {', '.join(anomaly.recommended_actions)}")
            if hasattr(anomaly, "model_reasoning") and len(anomaly.model_reasoning) > 100:
                print(f"\n📋 DETAILED ANALYSIS:\n{anomaly.model_reasoning[:100]}...")
    else:
        print("\n✅ No anomalies detected")


if __name__ == "__main__":
    asyncio.run(main())
