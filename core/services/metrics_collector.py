"""
Advanced metrics collection system demonstrating modern Python patterns.

Key patterns demonstrated:
- Protocol-based dependency injection
- Generic types for robust error handling
- Async context managers for resource lifecycle
- Structured concurrency with asyncio.TaskGroup
- Comprehensive error boundaries
"""

import asyncio
import random
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Generic, Protocol, TypeVar

import structlog
from pydantic import BaseModel, Field

from core.domain.models import MetricType, SystemMetric

# Configure structured logging (production-ready observability)
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Generic Result type for explicit error handling
ValueT = TypeVar("ValueT")
ErrorT = TypeVar("ErrorT", bound=BaseException, default=Exception)


class Result(Generic[ValueT, ErrorT]):
    """
    Explicit error handling without exceptions for expected failures.

    Why: Makes error paths visible in type system, forces handling decisions.
    When to use: When failure is expected business logic, not exceptional.
    """

    def __init__(self, value: ValueT | None = None, error: ErrorT | None = None) -> None:
        if value is not None and error is not None:
            raise ValueError("Result cannot have both value and error")
        if value is None and error is None:
            raise ValueError("Result must have either value or error")
        self._value: ValueT | None = value
        self._error: ErrorT | None = error

    @classmethod
    def ok(cls, value: ValueT) -> "Result[ValueT, ErrorT]":
        return cls(value=value)

    @classmethod
    def err(cls, error: ErrorT) -> "Result[ValueT, ErrorT]":
        return cls(error=error)

    def is_ok(self) -> bool:
        return self._error is None

    def is_err(self) -> bool:
        return self._error is not None

    def unwrap(self) -> ValueT:
        if self._error:
            raise self._error
        return self._value  # type: ignore

    def unwrap_or(self, default: ValueT) -> ValueT:
        return self._value if self._error is None else default  # type: ignore

    def unwrap_err(self) -> ErrorT:
        if self._error is None:
            raise ValueError("Called unwrap_err() on an Ok value")
        return self._error


class MetricsSource(Protocol):
    """
    Protocol defining how to collect metrics from different sources.

    Why Protocol over ABC: Structural typing, easier mocking, less coupling.
    Design: Single method, focused responsibility, async-first.
    """

    source_name: str

    async def collect_metrics(self) -> Result[list[SystemMetric]]:
        """
        Collect metrics from the source.

        Returns:
            Result[list[SystemMetric]]: Result containing the collected metrics or an exception.
        """
        ...


class SystemMetricsSource:
    """
    Simulated system metrics collection.

    In production: This would integrate with Prometheus, CloudWatch, DataDog, etc.
    Design principles: Fail fast on configuration, graceful degradation on runtime errors.
    """

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name
        self.logger = logger.bind(source=source_name)

    async def collect_metrics(self) -> Result[list[SystemMetric]]:
        """
        Simulate collecting system metrics with realistic error conditions.

        Returns:
            Result[list[SystemMetric]]: Result containing the collected metrics or an exception.
        """

        try:
            # Simulate network delay and occasional failures
            await asyncio.sleep(random.uniform(0.1, 0.5))

            # Simulate 5% failure rate (realistic for monitoring systems)
            if random.random() < 0.05:
                raise ConnectionError(f"Failed to connect to {self.source_name}")

            # Generate realistic metric data
            now = datetime.now(UTC)
            metrics = [
                SystemMetric(
                    metric_type=MetricType.CPU,
                    value=random.uniform(10, 90),
                    unit="percent",
                    timestamp=now,
                    source=self.source_name,
                    tags={"environment": "production", "region": "us-east-1"},
                ),
                SystemMetric(
                    metric_type=MetricType.MEMORY,
                    value=random.uniform(30, 85),
                    unit="percent",
                    timestamp=now,
                    source=self.source_name,
                    tags={"environment": "production", "region": "us-east-1"},
                ),
                SystemMetric(
                    metric_type=MetricType.DISK_IO,
                    value=random.uniform(40, 90),
                    unit="percent",
                    timestamp=now,
                    source=self.source_name,
                    tags={"environment": "production", "region": "us-east-1"},
                ),
                SystemMetric(
                    metric_type=MetricType.ERROR_RATE,
                    value=random.uniform(0, 10),
                    unit="errors_per_minute",
                    timestamp=now,
                    source=self.source_name,
                    tags={"service": "api", "version": "v1.2.3"},
                ),
            ]

            self.logger.info("metrics_collected", count=len(metrics))
            return Result.ok(metrics)

        except Exception as e:
            self.logger.exception("metrics_collection_failed", error=str(e))
            return Result.err(e)


class MetricsCollectorConfig(BaseModel):
    """
    Configuration with validation and smart defaults.
    """

    collection_interval_seconds: float = Field(
        default=30.0,
        gt=1.0,
        description="Interval between metric collections in seconds.",
    )
    max_concurrent_sources: int = Field(
        default=10,
        gt=0,
        description="Max number of concurrent metric collections.",
    )
    timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        description="Timeout for individual metric collection in seconds.",
    )
    retry_attempts: int = Field(
        default=3,
        gt=0,
        description="Number of retry attempts for failed metric collections.",
    )


class MetricsCollector:
    """
    Orchestrates metric collection from multiple sources with proper error handling.

    Design principles:
    - Fail fast on startup (configuration validation)
    - Graceful degradation during runtime (partial failures OK)
    - Observable (structured logging for debugging)
    - Resource-aware (connection pooling, timeouts, backpressure)
    """

    def __init__(self, config: MetricsCollectorConfig) -> None:
        self.config = config
        self.sources: list[MetricsSource] = []
        self.logger = logger.bind(component="metrics_collector")
        self._is_running: bool = False

    def add_source(self, source: MetricsSource) -> None:
        """Add a metrics source. Validates source implements protocol correctly."""
        if not hasattr(source, "collect_metrics"):
            raise TypeError(f"Source {source} must implement MetricsSource protocol")
        self.sources.append(source)
        self.logger.info("source_added", source_type=type(source).__name__)

    def remove_source(self, source: MetricsSource) -> None:
        """Remove a metrics source."""
        self.sources.remove(source)
        self.logger.info("source_removed", source_type=type(source).__name__)

    @asynccontextmanager
    async def collection_session(self) -> AsyncIterator["MetricsCollector"]:
        """
        Async context manager for proper resource lifecycle.

        Why: Ensures cleanup happens even if exceptions occur.
        Pattern: Acquire resources in __aenter__, release in __aexit__.
        """
        self.logger.info("metrics_collection_session_started")
        self._is_running = True

        try:
            await asyncio.sleep(self.config.collection_interval_seconds)
            yield self
        finally:
            self._is_running = False
            self.logger.info("metrics_collection_session_ended")

    async def collect_once(self) -> Result[list[SystemMetric]]:
        """
        Collect metrics from all sources with structured concurrency.

        Key pattern: Use TaskGroup for structured concurrency (Python 3.11+).
        Why: Automatic cleanup, proper exception handling, no orphaned tasks.
        """
        if not self._is_running:
            raise RuntimeError("Collector not running - use collection_session()")

        start_time = time.perf_counter()
        collected_metrics: list[SystemMetric] = []

        # Structured concurrency - all tasks managed together
        async with asyncio.TaskGroup() as task_group:
            # Create tasks for each source with timeout
            tasks = [
                task_group.create_task(
                    asyncio.wait_for(
                        source.collect_metrics(),
                        timeout=self.config.timeout_seconds,
                    )
                )
                for source in self.sources
            ]

        # Process results (TaskGroup ensures all tasks completed or cancelled)
        successful_collections = 0
        for task in tasks:
            try:
                result = task.result()
                if result.is_ok():
                    collected_metrics.extend(result.unwrap())
                    successful_collections += 1
                else:
                    # Log error but continue processing other sources
                    self.logger.warning(
                        "source_collection_failed",
                        error=str(result.unwrap_err()),
                        source=task.get_name(),
                    )
            except TimeoutError:
                self.logger.warning(
                    "source_collection_timeout",
                    source=task.get_name(),
                )
            except Exception as e:
                self.logger.exception(
                    "unexpected_source_collection_error",
                    error=str(e),
                    source=task.get_name(),
                )

        duration_time = time.perf_counter() - start_time

        self.logger.info(
            "metrics_collection_completed",
            total_metrics=len(collected_metrics),
            successful_sources=successful_collections,
            total_sources=len(self.sources),
            duration_seconds=round(duration_time, 3),
        )
        return Result.ok(collected_metrics)

    async def collect_continuously(self) -> AsyncIterator[list[SystemMetric]]:
        """
        Continuous collection with backpressure handling.

        Why generator: Memory efficient, backpressure-aware, composable.
        Pattern: Yield results as available, handle timing internally.
        """
        self.logger.info(
            "metrics_collection_started", interval_seconds=self.config.collection_interval_seconds
        )

        while self._is_running:
            collection_start_time = time.perf_counter()

            try:
                metrics = await self.collect_once()
                yield metrics.unwrap()

                elapsed_time = time.perf_counter() - collection_start_time
                sleep_time = max(0, self.config.collection_interval_seconds - elapsed_time)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    self.logger.warning(
                        "metrics_collection_slower_than_interval",
                        elapsed_seconds=round(elapsed_time, 3),
                        interval_seconds=self.config.collection_interval_seconds,
                    )

            except Exception as e:
                self.logger.exception(
                    "unexpected_continuous_metrics_collection_error", error=str(e)
                )
                # Exponential backoff on errors
                await asyncio.sleep(min(60, self.config.collection_interval_seconds * 2))


async def main() -> None:
    """Demonstrate usage of the metrics collector system"""

    # configuration with validation
    config = MetricsCollectorConfig(
        collection_interval_seconds=5.0,
        max_concurrent_sources=5,
        timeout_seconds=3.0,
        retry_attempts=2,
    )

    # create metrics collector
    metrics_collector = MetricsCollector(config)

    # add multiple sources (simulating different services)
    for service in ["web-api", "database", "cache", "queue"]:
        metrics_collector.add_source(SystemMetricsSource(f"{service}-prod"))

    # proper resource management with context manager
    async with metrics_collector.collection_session():
        # single collection example
        metrics = await metrics_collector.collect_once()
        if metrics.is_ok():
            print("Metrics collected successfully:", metrics.unwrap())
        else:
            print("Failed to collect metrics:", metrics.unwrap_err())

    # Continuous collection with explicit limit for demo
    collection_count = 0
    async for batch in metrics_collector.collect_continuously():
        print(f"Batch {collection_count + 1}: {len(batch)} metrics")
        collection_count += 1

        if collection_count >= 3:  # Stop after 3 collections for demo
            break

    # add a new source
    metrics_collector.add_source(SystemMetricsSource("new-service-prod"))
    print("New source added:", metrics_collector.sources[-1].source_name)

    # remove new source
    metrics_collector.remove_source(metrics_collector.sources[-1])
    print("New source removed:", metrics_collector.sources[-1].source_name)


if __name__ == "__main__":
    asyncio.run(main())
