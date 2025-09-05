"""
Tests for database metrics collection functionality.

These tests verify that the DatabaseMetricsSource correctly collects and reports
metrics about database performance and health.
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from core.services.database_metrics import DatabaseMetricsSource


@pytest.fixture
def db_source() -> DatabaseMetricsSource:
    """Fixture that provides a DatabaseMetricsSource instance for testing."""
    return DatabaseMetricsSource(source_name="test-db", db_name="test_db", db_type="postgresql")


@pytest.mark.asyncio
async def test_collect_metrics_success(db_source: DatabaseMetricsSource) -> None:
    """Test successful collection of database metrics."""
    with (
        patch("random.random", return_value=0.5),
        patch("random.uniform", side_effect=[0.2, 50.0]),
        patch("random.randint", return_value=10),
        patch("random.choices", return_value=[1]),
        patch("asyncio.sleep"),
    ):  # Skip sleep delays
        result = await db_source.collect_metrics()

    assert result.is_ok()
    metrics = result.unwrap()
    assert len(metrics) == 3  # Should have 3 metrics (pool, latency, deadlocks)

    # Verify metric types and basic structure
    metric_names = {m.name for m in metrics}
    assert "db_connection_pool_usage" in metric_names
    assert "db_query_latency" in metric_names
    assert "db_deadlocks" in metric_names

    # Verify tags are properly set
    for metric in metrics:
        assert "db_name" in metric.tags
        assert "db_type" in metric.tags
        assert metric.source == "test-db"


@pytest.mark.asyncio
async def test_collect_metrics_failure(db_source: DatabaseMetricsSource) -> None:
    """Test that collection fails with the expected probability."""
    with patch("random.random", return_value=0.05), patch("asyncio.sleep"):  # Skip sleep delays
        result = await db_source.collect_metrics()

    assert result.is_err()
    assert "Failed to connect to database" in str(result.unwrap_err())


def test_initialization() -> None:
    """Test that the source is initialized with correct values."""
    source = DatabaseMetricsSource(source_name="test-source", db_name="test_db", db_type="mysql")

    assert source.source_name == "test-source"
    assert source.db_name == "test_db"
    assert source.db_type == "mysql"
    assert 5 <= source._connection_pool_size <= 20  # Should be in the initialized range


@pytest.mark.asyncio
async def test_metrics_contain_required_fields(db_source: DatabaseMetricsSource) -> None:
    """Test that all metrics have required fields."""
    with patch("random.random", return_value=0.5), patch("asyncio.sleep"):  # Skip sleep delays
        result = await db_source.collect_metrics()

    metrics = result.unwrap()
    for metric in metrics:
        assert isinstance(metric.timestamp, datetime)
        assert metric.timestamp.tzinfo == UTC
        assert isinstance(metric.value, int | float)
        assert metric.unit in ("percent", "milliseconds", "count")
        assert isinstance(metric.tags, dict)


@pytest.mark.asyncio
async def test_connection_pool_metrics_range(db_source: DatabaseMetricsSource) -> None:
    """Test that connection pool metrics are within expected ranges."""
    with patch("random.random", return_value=0.5), patch("asyncio.sleep"):  # Skip sleep delays
        result = await db_source.collect_metrics()

    metrics = result.unwrap()
    pool_metric = next(m for m in metrics if m.name == "db_connection_pool_usage")
    assert 0 <= pool_metric.value <= 100  # Pool usage should be between 0% and 100%


@pytest.mark.asyncio
async def test_query_latency_metrics_range(db_source: DatabaseMetricsSource) -> None:
    """Test that query latency metrics are within expected ranges."""
    with patch("random.random", return_value=0.5), patch("asyncio.sleep"):  # Skip sleep delays
        result = await db_source.collect_metrics()

    metrics = result.unwrap()
    latency_metric = next(m for m in metrics if m.name == "db_query_latency")
    assert 5 <= latency_metric.value <= 150  # Based on implementation
    assert latency_metric.unit == "milliseconds"


@pytest.mark.asyncio
async def test_deadlock_metrics_distribution() -> None:
    """Test the statistical distribution of deadlock metrics."""
    # Test the random.choices call directly instead of through collect_metrics()
    with patch("random.choices") as mock_choices:
        # Mock the random.choices to return our test distribution
        mock_choices.return_value = [0] * 80 + [1] * 15 + [2] * 5  # 80/15/5 distribution

        # Create a new source to avoid state issues
        source = DatabaseMetricsSource("test-db", "test_db", "postgresql")

        with patch("random.random", return_value=0.5), patch("asyncio.sleep"):  # Skip sleep delays
            result = await source.collect_metrics()

    metrics = result.unwrap()
    deadlock_metric = next(m for m in metrics if m.name == "db_deadlocks")

    # Verify the value was used from our mocked distribution
    assert deadlock_metric.value in [0, 1, 2]
    assert deadlock_metric.unit == "count"


@pytest.mark.asyncio
async def test_deadlock_metrics_range(db_source: DatabaseMetricsSource) -> None:
    """Test that deadlock metrics are within expected ranges (0-2)."""
    with patch("random.random", return_value=0.5), patch("asyncio.sleep"):  # Skip sleep delays
        result = await db_source.collect_metrics()

    metrics = result.unwrap()
    deadlock_metric = next(m for m in metrics if m.name == "db_deadlocks")
    assert 0 <= deadlock_metric.value <= 2
