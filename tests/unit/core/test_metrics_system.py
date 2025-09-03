"""
Test suite demonstrating modern Python testing patterns.

Testing philosophy:
- Fast feedback (unit tests run in milliseconds)
- Property-based testing for edge cases
- Clear test names that describe behavior
- Minimal mocking (test real behavior when possible)
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.domain.models import AnomalyDetection, MetricType, Severity, SystemMetric
from core.services.metrics_collector import (
    MetricsCollector,
    MetricsCollectorConfig,
    Result,
    SystemMetricsSource,
)


class TestResult:
    """Test the Result type for explicit error handling."""

    def test_result_ok_creates_successful_result(self) -> None:
        result: Result[str, Exception] = Result.ok("success")
        assert result.is_ok()
        assert not result.is_err()
        assert result.unwrap() == "success"

    def test_result_error_creates_failed_result(self) -> None:
        error = ValueError("test error")
        result: Result[str, ValueError] = Result.err(error)
        assert not result.is_ok()
        assert result.is_err()
        assert result.unwrap_or("default") == "default"

    def test_unwrap_raises_on_error_result(self) -> None:
        error = ValueError("test error")
        result: Result[str, ValueError] = Result.err(error)

        with pytest.raises(ValueError, match="test error"):
            result.unwrap()


class TestSystemMetric:
    """Test domain models with property-based testing."""

    @given(value=st.floats(min_value=0.0, max_value=100.0), source=st.text(min_size=1, max_size=50))
    def test_system_metric_creation_with_random_data(self, value: float, source: str) -> None:
        """Property-based test: any valid inputs should create valid metric."""
        metric = SystemMetric(
            metric_type=MetricType.CPU, value=value, unit="percent", source=source
        )

        assert metric.value == value
        assert metric.source == source
        assert isinstance(metric.timestamp, datetime)
        assert metric.timestamp.tzinfo == UTC

    def test_metric_immutability(self) -> None:
        """Ensure metrics can't be accidentally modified."""
        metric = SystemMetric(
            metric_type=MetricType.MEMORY, value=50.0, unit="percent", source="test"
        )

        # This should raise an error due to frozen=True
        with pytest.raises(ValueError, match="frozen"):
            metric.value = 75.0  # type: ignore


class TestMetricsCollectorConfig:
    """Test configuration validation."""

    def test_valid_config_creation(self) -> None:
        config = MetricsCollectorConfig(
            collection_interval_seconds=30.0,
            max_concurrent_sources=5,
            timeout_seconds=10.0,
            retry_attempts=3,
        )

        assert config.collection_interval_seconds == 30.0
        assert config.max_concurrent_sources == 5

    def test_invalid_config_raises_validation_error(self) -> None:
        """Ensure configuration validation catches invalid values."""
        with pytest.raises(ValueError):
            MetricsCollectorConfig(collection_interval_seconds=-1.0)

        with pytest.raises(ValueError):
            MetricsCollectorConfig(max_concurrent_sources=0)


class TestSystemMetricsSource:
    """Test metrics source with realistic async scenarios."""

    @pytest.fixture
    def source(self) -> SystemMetricsSource:
        return SystemMetricsSource("test-service")

    async def test_successful_collection(self, source: SystemMetricsSource) -> None:
        """Test normal operation path."""
        result = await source.collect_metrics()

        assert result.is_ok()
        metrics = result.unwrap()
        assert len(metrics) > 0

        # Verify metric structure
        for metric in metrics:
            assert isinstance(metric, SystemMetric)
            assert metric.source == "test-service"
            assert metric.timestamp.tzinfo == UTC

    async def test_collection_creates_different_metrics_each_time(
        self, source: SystemMetricsSource, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure metrics are actually dynamic."""
        # Mock random.random() to always return a value that won't trigger the 5% failure rate
        monkeypatch.setattr("random.random", lambda: 0.1)

        result1 = await source.collect_metrics()
        result2 = await source.collect_metrics()

        assert result1.is_ok() and result2.is_ok(), "Both collections should succeed"

        metrics1 = result1.unwrap()
        metrics2 = result2.unwrap()

        # Find CPU metrics
        cpu_metric1 = next(m for m in metrics1 if m.metric_type == MetricType.CPU)
        cpu_metric2 = next(m for m in metrics2 if m.metric_type == MetricType.CPU)

        # Verify the values are different (since they should be randomly generated)
        assert cpu_metric1.value != cpu_metric2.value, (
            "CPU values should be different between collections"
        )


class MockMetricsSource:
    """Test double that implements MetricsSource protocol."""

    def __init__(
        self,
        should_fail: bool = False,
        delay_seconds: float = 0.0,
        source_name: str = "mock-source",
    ) -> None:
        self.should_fail = should_fail
        self.delay_seconds = delay_seconds
        self.source_name = source_name
        self.call_count = 0

    async def collect_metrics(self) -> Result[list[SystemMetric], Exception]:
        self.call_count += 1

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.should_fail:
            return Result.err(ConnectionError("Mock failure"))

        return Result.ok(
            [
                SystemMetric(
                    metric_type=MetricType.CPU, value=50.0, unit="percent", source=self.source_name
                )
            ]
        )


class TestMetricsCollector:
    """Test the metrics collector with various scenarios."""

    @pytest.fixture
    def config(self) -> MetricsCollectorConfig:
        """Create a test configuration."""
        return MetricsCollectorConfig(
            collection_interval_seconds=2.0,
            timeout_seconds=2.0,
            retry_attempts=1,
            max_concurrent_sources=10,
        )

    @pytest.fixture
    async def collector(self, config: MetricsCollectorConfig) -> AsyncIterator[MetricsCollector]:
        """Create a test collector with a clean state for each test."""
        collector = MetricsCollector(config)
        yield collector
        # Cleanup if needed

    @pytest.mark.parametrize("success_count,fail_count", [(2, 1), (0, 2), (3, 0)])
    @pytest.mark.asyncio
    async def test_collect_once_with_mixed_sources(
        self,
        collector: MetricsCollector,
        success_count: int,
        fail_count: int,
    ) -> None:
        """Test collection with a mix of successful and failing sources."""
        # Add successful sources
        for _ in range(success_count):
            collector.add_source(MockMetricsSource(should_fail=False))

        # Add failing sources
        for _ in range(fail_count):
            collector.add_source(MockMetricsSource(should_fail=True))

        async with collector.collection_session():
            result = await collector.collect_once()

        # Verify we got metrics from successful sources
        if success_count > 0:
            assert result.is_ok()
            metrics = result.unwrap()
            assert len(metrics) == success_count
            assert all(isinstance(m, SystemMetric) for m in metrics)
        else:
            assert result.is_err()

    @pytest.mark.asyncio
    async def test_continuous_collection_stops_gracefully(
        self, collector: MetricsCollector
    ) -> None:
        """Test continuous collection with proper shutdown."""
        collector.add_source(MockMetricsSource())
        collected_batches: list[list[SystemMetric]] = []

        async def process_batch(batch: list[SystemMetric]) -> None:
            collected_batches.append(batch)

        async with collector.collection_session():
            # Collect a few batches then break (simulates graceful shutdown)
            async for batch in collector.collect_continuously():
                await process_batch(batch)
                if len(collected_batches) >= 2:
                    break

        assert len(collected_batches) == 2
        assert all(len(batch) > 0 for batch in collected_batches)

    @pytest.mark.parametrize("source_count", [1, 5, 10])
    @pytest.mark.asyncio
    async def test_concurrent_source_execution(
        self, collector: MetricsCollector, source_count: int
    ) -> None:
        """Test that sources are collected from concurrently."""
        # Add sources with sequential delays
        for i in range(source_count):
            collector.add_source(
                MockMetricsSource(
                    delay_seconds=0.1,
                    source_name=f"source-{i}",
                )
            )

        async with collector.collection_session():
            start_time = asyncio.get_event_loop().time()
            result = await collector.collect_once()
            duration = asyncio.get_event_loop().time() - start_time

        # All sources should complete in roughly the same time as one source
        # due to concurrent execution
        assert duration < 0.5, f"Expected concurrent execution, took {duration:.2f}s"
        if source_count > 0:
            assert result.is_ok()
            metrics = result.unwrap()
            assert len(metrics) == source_count
            assert all(isinstance(m, SystemMetric) for m in metrics)


# Integration test demonstrating realistic usage
class TestMetricsCollectionIntegration:
    """End-to-end style tests with realistic configurations."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self) -> None:
        """Test a complete metrics collection and processing workflow."""
        # Configure with realistic settings
        config = MetricsCollectorConfig(
            collection_interval_seconds=2.0,
            timeout_seconds=5.0,
            retry_attempts=2,
            max_concurrent_sources=5,
        )

        collector = MetricsCollector(config)

        # Add multiple realistic sources with different characteristics
        collector.add_source(MockMetricsSource(source_name="fast-source"))
        collector.add_source(
            MockMetricsSource(
                delay_seconds=0.1,
                source_name="medium-source",
            )
        )
        collector.add_source(
            MockMetricsSource(
                delay_seconds=0.2,
                source_name="slow-source",
            )
        )

        # Test collection
        async with collector.collection_session():
            # Single collection
            result = await collector.collect_once()
            assert result.is_ok()
            metrics = result.unwrap()
            assert len(metrics) == 3  # All sources should succeed

            # Continuous collection
            collected: list[list[SystemMetric]] = []
            async for batch in collector.collect_continuously():
                collected.append(batch)
                if len(collected) >= 2:
                    break

            assert len(collected) == 2
            assert all(len(batch) == 3 for batch in collected)


class TestPerformanceRegression:
    """Performance regression tests with realistic baselines."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_collection_performance_baseline(self) -> None:
        """Establish performance baseline for metrics collection."""
        config = MetricsCollectorConfig(
            collection_interval_seconds=2.0,
            timeout_seconds=1.0,
            retry_attempts=1,
            max_concurrent_sources=10,
        )

        collector = MetricsCollector(config)

        # Add multiple fast sources
        for i in range(5):
            collector.add_source(
                MockMetricsSource(
                    delay_seconds=0.01,
                    source_name=f"perf-source-{i}",
                )
            )

        # Measure collection time
        async with collector.collection_session():
            start_time = asyncio.get_event_loop().time()
            result = await collector.collect_once()
            duration = asyncio.get_event_loop().time() - start_time

        # Verify performance meets expectations
        assert result.is_ok()
        assert duration < 0.5, f"Collection took too long: {duration:.3f}s"
        metrics = result.unwrap()
        assert len(metrics) == 5
        assert all(m.value == 50.0 for m in metrics)  # Verify mock data


# Fixtures for reusable test data
@pytest.fixture
def sample_metrics() -> list[SystemMetric]:
    """Generate realistic test metrics."""
    now = datetime.now(UTC)
    return [
        SystemMetric(
            metric_type=MetricType.CPU,
            value=75.5,
            unit="percent",
            timestamp=now,
            source="web-server-1",
            tags={"environment": "prod", "region": "us-east-1"},
        ),
        SystemMetric(
            metric_type=MetricType.MEMORY,
            value=82.3,
            unit="percent",
            timestamp=now,
            source="web-server-1",
            tags={"environment": "prod", "region": "us-east-1"},
        ),
        SystemMetric(
            metric_type=MetricType.ERROR_RATE,
            value=15.7,
            unit="errors_per_minute",
            timestamp=now,
            source="api-service",
            tags={"service": "user-api", "version": "v2.1.0"},
        ),
    ]


@pytest.fixture
def sample_anomaly() -> AnomalyDetection:
    """Generate realistic anomaly detection result."""
    return AnomalyDetection(
        severity=Severity.HIGH,
        confidence=0.87,
        affected_metrics=[MetricType.CPU, MetricType.ERROR_RATE],
        root_cause_hypothesis=(
            "High CPU usage correlates with increased error rate, suggesting "
            "resource exhaustion in web server pool"
        ),
        recommended_actions=[
            "Scale web server instances horizontally",
            "Investigate recent deployments for performance regressions",
            "Check database connection pool saturation",
        ],
        correlation_window_minutes=15,
        model_reasoning=(
            "Observed 3x normal CPU usage coinciding with 5x error rate "
            "increase over 15-minute window"
        ),
    )


class TestDomainModels:
    """Test domain model validation and business logic."""

    def test_system_metric_validation(self) -> None:
        """Test that invalid metrics are rejected."""
        # Valid metric should work
        metric = SystemMetric(metric_type=MetricType.CPU, value=50.0, unit="percent", source="test")
        assert metric.value == 50.0

        # Test that non-float values are rejected by Pydantic validation
        with pytest.raises(
            ValueError
        ):  # Pydantic raises ValidationError which is a subclass of ValueError
            SystemMetric(
                metric_type=MetricType.CPU,
                value="not a number",  # type: ignore[arg-type]
                unit="percent",
                source="test",
            )

    def test_anomaly_detection_confidence_bounds(self) -> None:
        """Test that confidence is properly bounded."""
        # Valid confidence
        anomaly = AnomalyDetection(
            severity=Severity.LOW,
            confidence=0.75,
            affected_metrics=[MetricType.CPU],
            root_cause_hypothesis="Test hypothesis",
            recommended_actions=["Test action"],
            correlation_window_minutes=10,
            model_reasoning="Test reasoning",
        )
        assert anomaly.confidence == 0.75

        # Invalid confidence should be rejected
        with pytest.raises(ValueError):
            AnomalyDetection(
                severity=Severity.LOW,
                confidence=1.5,  # > 1.0, should fail
                affected_metrics=[MetricType.CPU],
                root_cause_hypothesis="Test",
                recommended_actions=["Test"],
                correlation_window_minutes=10,
                model_reasoning="Test",
            )


# Running the tests
if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
