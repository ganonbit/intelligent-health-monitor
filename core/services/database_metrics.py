"""
Database metrics collection implementation.

This module provides a DatabaseMetricsSource class that implements the MetricsSource
protocol for collecting database-related metrics like connection pool usage,
query latency, and deadlocks.
"""

import asyncio
import random
from datetime import UTC, datetime

from core.domain.models import MetricType, SystemMetric
from core.services.metrics_collector import Result, logger


class DatabaseMetricsSource:
    """
    Simulated database metrics collection.

    Tracks key database metrics like connection pool usage, query latency, and deadlocks.
    Has a 10% failure rate to simulate database flakiness.
    """

    def __init__(self, source_name: str, db_name: str, db_type: str = "postgresql") -> None:
        """Initialize the database metrics source.

        Args:
            source_name: Name of the metrics source
            db_name: Name of the database being monitored
            db_type: Type of database (e.g., postgresql, mysql)
        """
        self.source_name = source_name
        self.db_name = db_name
        self.db_type = db_type
        self.logger = logger.bind(source=source_name, db_name=db_name, db_type=db_type)
        self._connection_pool_size = random.randint(5, 20)  # Simulate dynamic pool size
        self._tables = ["users", "orders", "products", "inventory"]  # Example tables

    async def collect_metrics(self) -> Result[list[SystemMetric], Exception]:
        """
        Simulate collecting database metrics with realistic error conditions.

        Returns:
            Result[list[SystemMetric]]: Result containing the collected metrics or an exception.
        """
        try:
            # Simulate network/database latency
            await asyncio.sleep(random.uniform(0.1, 1.0))

            # 10% chance of failure (higher than system metrics)
            if random.random() < 0.10:
                raise ConnectionError(f"Failed to connect to database {self.db_name}")

            now = datetime.now(UTC)
            base_tags = {
                "environment": "production",
                "region": "us-east-1",
                "db_name": self.db_name,
                "db_type": self.db_type,
            }

            # Simulate connection pool metrics
            active_connections = random.randint(1, self._connection_pool_size)
            connection_pool_usage = (active_connections / self._connection_pool_size) * 100

            # Simulate query latency (in milliseconds)
            query_latency = random.uniform(5.0, 150.0)

            # Simulate deadlocks (0-2 per minute)
            deadlocks = random.choices([0, 1, 2], weights=[0.8, 0.15, 0.05])[0]

            # Update pool size occasionally
            if random.random() < 0.1:  # 10% chance to change pool size
                self._connection_pool_size = random.randint(5, 20)

            # Select a random table for this sample
            sample_table = random.choice(self._tables)

            metrics = [
                # Connection pool usage
                SystemMetric(
                    metric_type=MetricType.CUSTOM,
                    name="db_connection_pool_usage",
                    value=connection_pool_usage,
                    unit="percent",
                    timestamp=now,
                    source=self.source_name,
                    tags={**base_tags, "metric_type": "connection_pool", "pool_type": "main"},
                ),
                # Query latency
                SystemMetric(
                    metric_type=MetricType.RESPONSE_TIME,
                    name="db_query_latency",
                    value=query_latency,
                    unit="milliseconds",
                    timestamp=now,
                    source=self.source_name,
                    tags={**base_tags, "query_type": "select", "table": sample_table},
                ),
                # Deadlocks
                SystemMetric(
                    metric_type=MetricType.CUSTOM,
                    name="db_deadlocks",
                    value=deadlocks,
                    unit="count",
                    timestamp=now,
                    source=self.source_name,
                    tags={**base_tags, "metric_type": "deadlocks"},
                ),
            ]

            self.logger.info(
                "database_metrics_collected",
                count=len(metrics),
                db_name=self.db_name,
                active_connections=active_connections,
                pool_size=self._connection_pool_size,
            )
            return Result.ok(metrics)

        except Exception as e:
            self.logger.error("database_metrics_collection_failed", error=str(e))
            return Result.err(e)
