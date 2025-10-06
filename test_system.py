"""
Complete system test demonstrating the full monitoring pipeline.

This script tests:
1. Configuration loading and validation
2. Metrics collection from multiple sources
3. AI analysis and anomaly detection
4. Alert generation and dispatch
5. Error handling and fallbacks

Run with: uv run python test_system.py
"""

import asyncio
import os
from datetime import UTC, datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.config import get_config, print_config_summary, validate_config
from core.domain.models import MetricType, SystemMetric
from core.services.ai_analysis import SystemContext
from core.services.integrated_monitoring import IntegratedMonitoringService

console = Console()


class TestMetricsSource:
    """Test metrics source that generates specific scenarios for testing."""

    def __init__(self, source_name: str, scenario: str = "normal") -> None:
        self.source_name = source_name
        self.scenario = scenario

    async def collect_metrics(self):
        """Generate test metrics based on scenario."""
        from core.services.metrics_collector import Result

        try:
            await asyncio.sleep(0.1)  # Simulate collection time

            now = datetime.now(UTC)

            if self.scenario == "normal":
                # Normal operating metrics
                metrics = [
                    SystemMetric(
                        metric_type=MetricType.CPU,
                        value=45.0,
                        unit="percent",
                        timestamp=now,
                        source=self.source_name,
                    ),
                    SystemMetric(
                        metric_type=MetricType.MEMORY,
                        value=62.0,
                        unit="percent",
                        timestamp=now,
                        source=self.source_name,
                    ),
                    SystemMetric(
                        metric_type=MetricType.ERROR_RATE,
                        value=1.2,
                        unit="errors_per_minute",
                        timestamp=now,
                        source=self.source_name,
                    ),
                ]

            elif self.scenario == "high_cpu_errors":
                # Problematic scenario: High CPU correlates with high errors
                metrics = [
                    SystemMetric(
                        metric_type=MetricType.CPU,
                        value=87.5,  # Very high
                        unit="percent",
                        timestamp=now,
                        source=self.source_name,
                    ),
                    SystemMetric(
                        metric_type=MetricType.MEMORY,
                        value=58.0,  # Normal
                        unit="percent",
                        timestamp=now,
                        source=self.source_name,
                    ),
                    SystemMetric(
                        metric_type=MetricType.ERROR_RATE,
                        value=23.7,  # Very high, correlates with CPU
                        unit="errors_per_minute",
                        timestamp=now,
                        source=self.source_name,
                    ),
                ]

            elif self.scenario == "memory_leak":
                # Memory leak scenario: High memory, normal everything else
                metrics = [
                    SystemMetric(
                        metric_type=MetricType.CPU,
                        value=42.0,  # Normal
                        unit="percent",
                        timestamp=now,
                        source=self.source_name,
                    ),
                    SystemMetric(
                        metric_type=MetricType.MEMORY,
                        value=94.3,  # Critical
                        unit="percent",
                        timestamp=now,
                        source=self.source_name,
                    ),
                    SystemMetric(
                        metric_type=MetricType.ERROR_RATE,
                        value=1.8,  # Slightly elevated
                        unit="errors_per_minute",
                        timestamp=now,
                        source=self.source_name,
                    ),
                ]

            return Result.ok(metrics)

        except Exception as e:
            return Result.err(e)


async def test_configuration() -> bool:
    """Test configuration loading and validation."""

    console.print(Panel("🔧 Testing Configuration", style="blue"))

    try:
        # Test config loading
        validate_config()
        config = get_config()

        # Check required values
        if (
            not config.ai_provider.openai_api_key
            or config.ai_provider.openai_api_key == "your-openai-api-key-here"
        ):
            console.print(
                "❌ OpenAI API key not configured. Please set OPENAI_API_KEY in .env", style="red"
            )
            console.print("Get your key from: https://platform.openai.com/api-keys", style="yellow")
            return False

        console.print("✅ Configuration loaded successfully", style="green")
        print_config_summary()
        return True

    except Exception as e:
        console.print(f"❌ Configuration test failed: {e}", style="red")
        return False


async def test_metrics_collection() -> bool:
    """Test metrics collection from multiple sources."""

    console.print(Panel("📊 Testing Metrics Collection", style="blue"))

    try:
        from core.services.metrics_collector import MetricsCollector, MetricsCollectorConfig

        # Create collector
        config = MetricsCollectorConfig(timeout_seconds=5.0)
        collector = MetricsCollector(config)

        # Add test sources
        collector.add_source(TestMetricsSource("test-web-1", "normal"))
        collector.add_source(TestMetricsSource("test-db-1", "high_cpu_errors"))
        collector.add_source(TestMetricsSource("test-cache-1", "memory_leak"))

        # Test collection
        async with collector.collection_session():
            metrics_result = await collector.collect_once()

        if metrics_result.is_err():
            raise metrics_result.unwrap_err()

        metrics = metrics_result.unwrap()

        console.print(
            f"✅ Collected {len(metrics)} metrics from {len({m.source for m in metrics})} sources",
            style="green",
        )

        # Display metrics table
        table = Table(title="Collected Metrics")
        table.add_column("Source", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")

        for metric in metrics[:10]:  # Show first 10
            table.add_row(
                metric.source, metric.metric_type.value, f"{metric.value:.1f}", metric.unit
            )

        console.print(table)
        return True

    except Exception as e:
        console.print(f"❌ Metrics collection test failed: {e}", style="red")
        return False


async def test_ai_analysis() -> bool:
    """Test AI analysis with problematic metrics."""

    console.print(Panel("🤖 Testing AI Analysis", style="blue"))

    try:
        from core.services.ai_analysis import (
            AIAnalysisConfig,
            IntelligentMonitoringService,
            SystemContext,
        )

        # Create AI service
        config = AIAnalysisConfig(
            anomaly_threshold=0.6,  # Lower threshold for testing
            timeout_seconds=30.0,
        )
        ai_service = IntelligentMonitoringService(config)

        # Create problematic metrics
        now = datetime.now(UTC)
        problematic_metrics = [
            SystemMetric(
                metric_type=MetricType.CPU,
                value=89.2,
                unit="percent",
                timestamp=now,
                source="test-web-server",
            ),
            SystemMetric(
                metric_type=MetricType.ERROR_RATE,
                value=18.5,
                unit="errors_per_minute",
                timestamp=now,
                source="test-web-server",
            ),
            SystemMetric(
                metric_type=MetricType.MEMORY,
                value=67.0,
                unit="percent",
                timestamp=now,
                source="test-web-server",
            ),
        ]

        # System context
        system_context = SystemContext(
            system_name="test-web-server",
            system_type="web-server",
            environment="testing",
            expected_load_pattern="business-hours",
            recent_changes=["deployed v2.1.0", "increased traffic routing"],
            baseline_metrics={"cpu": 45.0, "memory": 60.0, "error_rate": 2.0},
        )

        console.print("🔍 Analyzing metrics with AI...", style="yellow")

        # Run AI analysis
        health_report = await ai_service.analyze_system_health(problematic_metrics, system_context)

        console.print(
            f"✅ AI analysis completed in {health_report.analysis_duration_seconds:.2f}s",
            style="green",
        )

        # Display results
        console.print("\n🏥 Health Report:")
        console.print(
            f"Overall Status: {health_report.overall_status.upper()}",
            style="red" if health_report.overall_status == "critical" else "yellow",
        )

        if health_report.anomalies:
            for i, anomaly in enumerate(health_report.anomalies, 1):
                console.print(f"\n🚨 Anomaly #{i}:", style="red")
                console.print(f"  Severity: {anomaly.severity.value.upper()}")
                console.print(f"  Confidence: {anomaly.confidence:.1%}")
                console.print(f"  Affected: {', '.join(m.value for m in anomaly.affected_metrics)}")
                console.print(f"  Hypothesis: {anomaly.root_cause_hypothesis}")
                console.print(
                    f"  Actions: {', '.join(anomaly.recommended_actions[:2])}"
                )  # First 2 actions
        else:
            console.print("No anomalies detected", style="green")

        return True

    except Exception as e:
        console.print(f"❌ AI analysis test failed: {e}", style="red")
        if "api key" in str(e).lower():
            console.print("💡 Make sure OPENAI_API_KEY is set in your .env file", style="yellow")
        return False


async def test_integrated_system() -> bool:
    """Test the complete integrated monitoring system."""

    console.print(Panel("🚀 Testing Integrated System", style="blue"))

    try:
        # Create integrated service
        monitoring_service = IntegratedMonitoringService()

        # Add test systems with different scenarios
        test_systems = [
            ("web-server-prod", "web-server", "normal"),
            ("database-prod", "database", "high_cpu_errors"),
            ("cache-prod", "cache", "memory_leak"),
        ]

        for system_name, system_type, scenario in test_systems:
            metrics_source = TestMetricsSource(system_name, scenario)

            system_context = SystemContext(
                system_name=system_name,
                system_type=system_type,
                environment="production",
                expected_load_pattern="24x7",
                baseline_metrics={"cpu": 50.0, "memory": 65.0, "error_rate": 1.5},
            )

            monitoring_service.add_metrics_source(metrics_source, system_context)

        console.print(f"Added {len(test_systems)} systems to monitor", style="green")

        # Run monitoring cycle
        console.print("🔄 Running monitoring cycle...", style="yellow")
        health_report = await monitoring_service.run_monitoring_cycle()

        if health_report:
            console.print("✅ Monitoring cycle completed successfully", style="green")

            # Create summary table
            summary_table = Table(title="Monitoring Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="white")

            summary_table.add_row("Overall Status", health_report.overall_status.upper())
            summary_table.add_row(
                "Systems Monitored", str(len({m.source for m in health_report.metrics}))
            )
            summary_table.add_row("Total Metrics", str(len(health_report.metrics)))
            summary_table.add_row("Anomalies Detected", str(len(health_report.anomalies)))
            summary_table.add_row(
                "Analysis Duration", f"{health_report.analysis_duration_seconds:.2f}s"
            )

            console.print(summary_table)

            return True
        else:
            console.print("❌ Monitoring cycle returned no results", style="red")
            return False

    except Exception as e:
        console.print(f"❌ Integrated system test failed: {e}", style="red")
        return False


async def test_error_handling() -> bool:
    """Test error handling and fallbacks."""

    console.print(Panel("🛡️ Testing Error Handling", style="blue"))

    try:
        # Test with invalid API key to trigger fallback
        original_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "invalid-key-for-testing"

        # Create service with invalid config
        monitoring_service = IntegratedMonitoringService()

        # Add a test source
        metrics_source = TestMetricsSource("test-fallback", "normal")
        system_context = SystemContext(
            system_name="test-fallback",
            system_type="test",
            environment="testing",
            expected_load_pattern="testing",
        )

        monitoring_service.add_metrics_source(metrics_source, system_context)

        # This should fallback gracefully when AI fails
        console.print("🔄 Testing fallback behavior...", style="yellow")
        health_report = await monitoring_service.run_monitoring_cycle()

        # Restore original key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        from core.config import reset_config_cache

        reset_config_cache()

        if health_report:
            console.print("✅ Fallback behavior working correctly", style="green")
            console.print(f"Fallback status: {health_report.overall_status}", style="yellow")
            return True
        else:
            console.print("❌ Fallback test failed", style="red")
            return False

    except Exception as e:
        # Restore original key on error
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        from core.config import reset_config_cache

        reset_config_cache()
        console.print(f"❌ Error handling test failed: {e}", style="red")
        return False


async def run_all_tests() -> None:
    """Run all system tests."""

    console.print(Panel("🧪 Intelligent Health Monitor - System Tests", style="bold blue"))

    tests = [
        ("Configuration", test_configuration),
        ("Metrics Collection", test_metrics_collection),
        ("AI Analysis", test_ai_analysis),
        ("Integrated System", test_integrated_system),
        ("Error Handling", test_error_handling),
    ]

    results = []

    for test_name, test_func in tests:
        console.print(f"\n{'=' * 60}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            console.print("\n⏹️  Tests interrupted by user", style="yellow")
            break
        except Exception as e:
            console.print(f"❌ {test_name} failed with exception: {e}", style="red")
            results.append((test_name, False))

    # Summary
    console.print(f"\n{'=' * 60}")
    console.print(Panel("📋 Test Results Summary", style="bold"))

    summary_table = Table()
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Result", style="white")

    passed = 0
    for test_name, result in results:
        if result:
            summary_table.add_row(test_name, "✅ PASSED")
            passed += 1
        else:
            summary_table.add_row(test_name, "❌ FAILED")

    console.print(summary_table)

    console.print(f"\n🎯 Results: {passed}/{len(results)} tests passed")

    if passed == len(results):
        console.print("🎉 All tests passed! Your monitoring system is ready.", style="green")
    else:
        console.print("⚠️  Some tests failed. Check the logs above for details.", style="yellow")


if __name__ == "__main__":
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        console.print("\n👋 Tests stopped by user", style="yellow")
    except Exception as e:
        console.print(f"\n💥 Test suite failed: {e}", style="red")
