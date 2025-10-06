"""
Integration service that combines metrics collection with AI analysis.

This demonstrates the complete end-to-end monitoring pipeline:
1. Collect metrics from multiple sources
2. Analyze with AI for anomalies
3. Generate actionable alerts
4. Maintain system context and learning

Architecture pattern: Event-driven pipeline with circuit breakers
"""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from core.config import AppConfig, get_config
from core.domain.models import AnomalyDetection, HealthReport, SystemMetric
from core.services.ai_analysis import (
    AIAnalysisConfig,
    IntelligentMonitoringService,
    OverallStatus,
    SystemContext,
)
from core.services.metrics_collector import MetricsCollector, MetricsCollectorConfig, MetricsSource

logger = structlog.get_logger()


@dataclass
class AlertEvent:
    """Represents an alert that should be sent to external systems."""

    timestamp: datetime
    severity: str
    title: str
    description: str
    source_system: str
    anomaly: AnomalyDetection | None = None
    metrics: list[SystemMetric] | None = None


class CircuitBreakerState:
    """Simple circuit breaker for AI service calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state = "closed"  # closed, open, half-open

    def can_execute(self) -> bool:
        """Check if operation can execute based on circuit breaker state."""

        if self.state == "closed":
            return True

        if self.state == "open":
            if self.last_failure_time:
                time_since_failure = datetime.now(UTC) - self.last_failure_time
                if time_since_failure.seconds >= self.recovery_timeout:
                    self.state = "half-open"
                    return True
            return False

        if self.state == "half-open":
            return True

        return False

    def record_success(self) -> None:
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(UTC)

        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class AlertManager:
    """Manages alert generation and dispatching."""

    def __init__(self) -> None:
        self.alert_history: deque[AlertEvent] = deque(maxlen=1000)
        self.logger = logger.bind(component="alert_manager")

    async def process_health_report(
        self, health_report: HealthReport, system_context: SystemContext
    ) -> list[AlertEvent]:
        """Convert health report to actionable alerts."""

        alerts = []

        # Generate alerts for each anomaly
        for anomaly in health_report.anomalies:
            alert = AlertEvent(
                timestamp=datetime.now(UTC),
                severity=anomaly.severity.value,
                title=f"Anomaly detected in {system_context.system_name}",
                description=anomaly.root_cause_hypothesis,
                source_system=system_context.system_name,
                anomaly=anomaly,
                metrics=health_report.metrics,
            )

            alerts.append(alert)
            self.alert_history.append(alert)

            self.logger.info(
                "alert_generated",
                severity=anomaly.severity.value,
                system=system_context.system_name,
                confidence=anomaly.confidence,
            )

        # Generate system health summary alerts for critical status
        if health_report.overall_status == "critical":
            summary_alert = AlertEvent(
                timestamp=datetime.now(UTC),
                severity="critical",
                title=f"System health critical: {system_context.system_name}",
                description="Multiple anomalies detected requiring immediate attention",
                source_system=system_context.system_name,
                metrics=health_report.metrics,
            )

            alerts.append(summary_alert)
            self.alert_history.append(summary_alert)

        return alerts

    async def dispatch_alerts(
        self,
        alerts: list[AlertEvent],
        handlers: list[Callable[[AlertEvent], None]] | None = None,
    ) -> None:
        """Dispatch alerts to configured handlers (email, Slack, PagerDuty, etc.)."""

        if not alerts:
            return

        # Default console handler for development
        if not handlers:
            handlers = [self._console_alert_handler]

        for alert in alerts:
            for handler in handlers:
                try:
                    await handler(alert) if asyncio.iscoroutinefunction(handler) else handler(alert)
                except Exception as e:
                    self.logger.error(
                        "alert_dispatch_failed", error=str(e), alert_title=alert.title
                    )

    def _console_alert_handler(self, alert: AlertEvent) -> None:
        """Development alert handler that prints to console."""

        severity_emoji = {"low": "💡", "medium": "⚠️", "high": "🚨", "critical": "🔥"}

        emoji = severity_emoji.get(alert.severity, "📢")

        print(f"\n{emoji} ALERT - {alert.severity.upper()}")
        print(f"Title: {alert.title}")
        print(f"System: {alert.source_system}")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Description: {alert.description}")

        if alert.anomaly:
            print(f"Confidence: {alert.anomaly.confidence:.1%}")
            print(f"Affected Metrics: {', '.join(m.value for m in alert.anomaly.affected_metrics)}")
            print("Recommended Actions:")
            for action in alert.anomaly.recommended_actions:
                print(f"  • {action}")

        print("-" * 80)


class IntegratedMonitoringService:
    """
    Main service that orchestrates the complete monitoring pipeline.

    This is the production-ready service that combines:
    - Metrics collection from multiple sources
    - AI-powered analysis and anomaly detection
    - Alert generation and dispatch
    - System context management and learning
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.logger = logger.bind(component="integrated_monitoring")

        # Initialize subsystems
        self._init_metrics_collection()
        self._init_ai_analysis()
        self._init_alert_management()

        # Circuit breaker for AI service
        self.ai_circuit_breaker = CircuitBreakerState()

        # System context management
        self.system_contexts: dict[str, SystemContext] = {}

        # Service state
        self._is_running = False

    def _init_metrics_collection(self) -> None:
        """Initialize metrics collection subsystem."""

        collector_config = MetricsCollectorConfig(
            collection_interval_seconds=self.config.monitoring.collection_interval_seconds,
            max_concurrent_sources=self.config.monitoring.max_concurrent_sources,
            timeout_seconds=self.config.monitoring.collection_timeout_seconds,
        )

        self.metrics_collector = MetricsCollector(collector_config)
        self.logger.info("metrics_collection_initialized")

    def _init_ai_analysis(self) -> None:
        """Initialize AI analysis subsystem."""

        ai_config = AIAnalysisConfig(
            model_name=self.config.ai_provider.anomaly_detection_model,
            temperature=self.config.ai_provider.default_temperature,
            timeout_seconds=self.config.ai_provider.default_timeout_seconds,
            anomaly_threshold=self.config.monitoring.anomaly_confidence_threshold,
            correlation_window_minutes=self.config.monitoring.correlation_window_minutes,
        )

        self.ai_monitoring = IntelligentMonitoringService(ai_config)
        self.logger.info("ai_analysis_initialized")

    def _init_alert_management(self) -> None:
        """Initialize alert management subsystem."""
        self.alert_manager = AlertManager()
        self.logger.info("alert_management_initialized")

    def add_metrics_source(self, source: MetricsSource, system_context: SystemContext) -> None:
        """Add a metrics source with associated system context."""

        self.metrics_collector.add_source(source)
        self.system_contexts[system_context.system_name] = system_context

        self.logger.info(
            "metrics_source_added",
            system_name=system_context.system_name,
            system_type=system_context.system_type,
        )

    async def run_monitoring_cycle(self) -> HealthReport | None:
        """
        Execute one complete monitoring cycle:
        1. Collect metrics
        2. Analyze with AI
        3. Generate alerts
        4. Update system context
        """

        cycle_start = datetime.now(UTC)
        self.logger.info("monitoring_cycle_starting")

        try:
            had_errors = False
            # Step 1: Collect metrics from all sources
            async with self.metrics_collector.collection_session():
                metrics_result = await self.metrics_collector.collect_once()

            if metrics_result.is_err():
                self.logger.warning("no_metrics_collected", error=str(metrics_result.unwrap_err()))
                return None

            metrics = metrics_result.unwrap()

            # Step 2: Group metrics by source system for analysis
            metrics_by_system = self._group_metrics_by_system(metrics)

            # Step 3: Analyze each system separately
            health_reports = []
            all_alerts = []

            for system_name, system_metrics in metrics_by_system.items():
                system_context = self.system_contexts.get(
                    system_name,
                    SystemContext(
                        system_name=system_name,
                        system_type="unknown",
                        expected_load_pattern="unknown",
                    ),
                )

                # AI analysis with circuit breaker
                health_report = None
                if self.ai_circuit_breaker.can_execute():
                    try:
                        health_report = await self.ai_monitoring.analyze_system_health(
                            system_metrics, system_context
                        )
                        self.ai_circuit_breaker.record_success()

                    except Exception as e:
                        self.logger.error("ai_analysis_failed", error=str(e), system=system_name)
                        self.ai_circuit_breaker.record_failure()
                        had_errors = True
                else:
                    self.logger.warning("ai_analysis_circuit_open", system=system_name)
                    had_errors = True

                # Fallback: Create basic health report if AI failed
                if not health_report:
                    health_report = HealthReport(
                        overall_status="warning",  # Conservative fallback
                        metrics=system_metrics,
                        anomalies=[],
                        analysis_duration_seconds=0.001,
                    )

                health_reports.append(health_report)

                # Step 4: Generate and dispatch alerts
                alerts = await self.alert_manager.process_health_report(
                    health_report, system_context
                )

                if alerts:
                    await self.alert_manager.dispatch_alerts(alerts)
                    all_alerts.extend(alerts)

            # Step 5: Update system context with learnings
            await self._update_system_context(metrics_by_system, health_reports)

            cycle_duration = (datetime.now(UTC) - cycle_start).total_seconds()

            self.logger.info(
                "monitoring_cycle_completed",
                systems_analyzed=len(metrics_by_system),
                total_metrics=len(metrics),
                alerts_generated=len(all_alerts),
                duration_seconds=round(cycle_duration, 3),
                ai_circuit_state=self.ai_circuit_breaker.state,
                degraded=had_errors,
            )

            # Return combined health report, downgrading to warning if degraded
            combined = self._combine_health_reports(health_reports)
            if had_errors and combined.overall_status == "healthy":
                combined = HealthReport(
                    overall_status="warning",
                    metrics=combined.metrics,
                    anomalies=combined.anomalies,
                    analysis_duration_seconds=combined.analysis_duration_seconds,
                )
            return combined

        except Exception as e:
            self.logger.error("monitoring_cycle_failed", error=str(e))
            return None

    async def run_continuous_monitoring(self) -> AsyncIterator[HealthReport]:
        """
        Run continuous monitoring with intelligent scheduling.

        Yields health reports as they become available.
        """

        self.logger.info(
            "continuous_monitoring_starting",
            interval=self.config.monitoring.analysis_interval_seconds,
        )
        self._is_running = True

        try:
            while self._is_running:
                analysis_start = datetime.now(UTC)

                # Run monitoring cycle
                health_report = await self.run_monitoring_cycle()

                if health_report:
                    yield health_report

                # Intelligent scheduling - adjust based on system state
                elapsed = (datetime.now(UTC) - analysis_start).total_seconds()
                base_interval = self.config.monitoring.analysis_interval_seconds

                # Reduce interval if critical issues detected
                if health_report and health_report.overall_status == "critical":
                    interval = base_interval / 2  # More frequent monitoring when critical
                elif self.ai_circuit_breaker.state == "open":
                    interval = base_interval * 2  # Less frequent when AI is failing
                else:
                    interval = base_interval

                sleep_time = max(0, interval - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            self.logger.info("continuous_monitoring_cancelled")
            raise
        except Exception as e:
            self.logger.error("continuous_monitoring_failed", error=str(e))
            raise
        finally:
            self._is_running = False

    def _group_metrics_by_system(
        self, metrics: list[SystemMetric]
    ) -> dict[str, list[SystemMetric]]:
        """Group metrics by source system for individual analysis."""

        systems: dict[str, list[SystemMetric]] = {}
        for metric in metrics:
            # Extract system name from source (e.g., "web-server-1-prod" -> "web-server-1")
            system_name = metric.source.replace("-prod", "").replace("-staging", "")

            if system_name not in systems:
                systems[system_name] = []
            systems[system_name].append(metric)

        return systems

    async def _update_system_context(
        self, metrics_by_system: dict[str, list[SystemMetric]], health_reports: list[HealthReport]
    ) -> None:
        """Update system context based on recent observations."""

        # Simple baseline learning - in production this would be more sophisticated
        for system_name, metrics in metrics_by_system.items():
            context = self.system_contexts.get(system_name)
            if not context:
                continue

            # Update baseline metrics with rolling averages
            for metric in metrics:
                metric_key = metric.metric_type.value
                current_baseline = context.baseline_metrics.get(metric_key, metric.value)

                # Simple exponential moving average
                alpha = 0.1  # Learning rate
                new_baseline = alpha * metric.value + (1 - alpha) * current_baseline
                context.baseline_metrics[metric_key] = new_baseline

    def _combine_health_reports(self, reports: list[HealthReport]) -> HealthReport:
        """Combine multiple system health reports into overall system health."""

        if not reports:
            return HealthReport(
                overall_status="warning",
                metrics=[],
                anomalies=[],
                analysis_duration_seconds=0.001,
            )

        # Combine all metrics and anomalies
        all_metrics: list[SystemMetric] = []
        all_anomalies = []
        total_duration = 0.0

        for report in reports:
            all_metrics.extend(report.metrics)
            all_anomalies.extend(report.anomalies)
            total_duration += report.analysis_duration_seconds

        # Determine overall status (most severe wins)
        statuses = [report.overall_status for report in reports]
        if "critical" in statuses:
            overall_status: OverallStatus = "critical"
        elif "warning" in statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return HealthReport(
            overall_status=overall_status,
            metrics=all_metrics,
            anomalies=all_anomalies,
            analysis_duration_seconds=max(total_duration, 0.001),
        )

    async def stop(self) -> None:
        """Gracefully stop the monitoring service."""
        self.logger.info("stopping_monitoring_service")
        self._is_running = False


# Example usage and demonstration
async def main() -> None:
    """Demonstrate the complete integrated monitoring system."""

    from core.services.ai_analysis import SystemContext
    from core.services.metrics_collector import SystemMetricsSource

    print("🚀 Starting Integrated Monitoring System Demo")

    # Initialize the service
    monitoring_service = IntegratedMonitoringService()

    # Add some example systems to monitor
    systems_to_monitor = [
        ("web-api-cluster", "web-server"),
        ("database-primary", "database"),
        ("cache-redis", "cache"),
    ]

    for system_name, system_type in systems_to_monitor:
        # Create metrics source
        metrics_source = SystemMetricsSource(f"{system_name}-prod")

        # Create system context
        system_context = SystemContext(
            system_name=system_name,
            system_type=system_type,
            environment="production",
            expected_load_pattern="business-hours",
            baseline_metrics={"cpu": 45.0, "memory": 60.0, "error_rate": 2.0},
        )

        monitoring_service.add_metrics_source(metrics_source, system_context)

    # Run a few monitoring cycles to demonstrate
    print("\n📊 Running monitoring cycles...")

    try:
        cycle_count = 0
        async for health_report in monitoring_service.run_continuous_monitoring():
            cycle_count += 1

            print(f"\n📋 CYCLE #{cycle_count} RESULTS")
            print(f"Overall Status: {health_report.overall_status.upper()}")
            print(f"Systems Analyzed: {len({m.source for m in health_report.metrics})}")
            print(f"Total Metrics: {len(health_report.metrics)}")
            print(f"Anomalies Detected: {len(health_report.anomalies)}")
            print(f"Analysis Duration: {health_report.analysis_duration_seconds:.2f}s")

            # Stop after a few cycles for demo
            if cycle_count >= 3:
                break

    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    finally:
        await monitoring_service.stop()
        print("✅ Monitoring service stopped gracefully")


if __name__ == "__main__":
    asyncio.run(main())
