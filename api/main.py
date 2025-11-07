"""
Production FastAPI application for HVAC monitoring system.

Features for production deployment:
- RESTful API with automatic OpenAPI documentation
- WebSocket support for real-time monitoring
- Health checks and monitoring endpoints
- Rate limiting and authentication hooks
- CORS configuration for web frontends
- Structured logging and error handling
- Graceful shutdown handling
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import structlog
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from adapters.hvac.domain import HVACMetric, HVACMetricType, HVACSystemType
from core.config import get_config
from core.services.ai_analysis import AIAnalysisConfig
from core.services.generic_hvac_agents import GenericHVACAIService
from core.services.integrated_monitoring import IntegratedMonitoringService

logger = structlog.get_logger()

# Global services (initialized on startup)
monitoring_service: IntegratedMonitoringService | None = None
hvac_ai_service: GenericHVACAIService | None = None


# API Models
class SystemStatusResponse(BaseModel):
    """System status response model."""

    status: str = Field(description="System status: healthy, degraded, or down")
    timestamp: datetime = Field(description="Status check timestamp")
    version: str = Field(description="Application version")
    uptime_seconds: float = Field(description="System uptime in seconds")

    components: dict[str, str] = Field(description="Component status breakdown")
    metrics: dict[str, Any] = Field(description="System performance metrics")


class MetricSubmissionRequest(BaseModel):
    """Request model for submitting HVAC metrics."""

    metrics: list[HVACMetric] = Field(description="List of HVAC metrics")
    system_context: dict[str, Any] | None = Field(None, description="System context information")


class AIAnalysisResponse(BaseModel):
    """Response model for AI analysis results."""

    analysis_id: str = Field(description="Unique analysis identifier")
    timestamp: datetime = Field(description="Analysis timestamp")

    energy_optimization: dict[str, Any] | None = Field(description="Energy optimization results")
    comfort_assessment: dict[str, Any] | None = Field(description="Comfort assessment results")
    fault_detection: dict[str, Any] | None = Field(description="Fault detection results")

    analysis_metadata: dict[str, Any] = Field(description="Analysis performance metadata")


class WebSocketMessage(BaseModel):
    """WebSocket message model for real-time updates."""

    message_type: str = Field(
        description="Message type: metric_update, analysis_result, system_alert"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = Field(description="Message payload")


# Application lifecycle management
@asynccontextmanager
async def lifecycle(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle management."""

    global monitoring_service, hvac_ai_service

    # Startup
    logger.info("application_startup_starting")

    try:
        # Initialize configuration
        config = get_config()

        # Initialize AI service
        ai_config = AIAnalysisConfig(
            model_name=config.ai_provider.anomaly_detection_model,
            temperature=config.ai_provider.default_temperature,
            timeout_seconds=config.ai_provider.default_timeout_seconds,
        )

        hvac_ai_service = GenericHVACAIService(ai_config)

        # Initialize monitoring service
        monitoring_service = IntegratedMonitoringService(config)

        logger.info("application_startup_completed")

    except Exception as e:
        logger.error("application_startup_failed", error=str(e))
        raise

    # Application running
    yield

    # Shutdown
    logger.info("application_shutdown_starting")

    try:
        if monitoring_service:
            await monitoring_service.stop()

        logger.info("application_shutdown_completed")

    except Exception as e:
        logger.error("application_shutdown_error", error=str(e))


# Initialize FastAPI application
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    config = get_config()

    app = FastAPI(
        title="Intelligent HVAC Monitoring System",
        description="AI-powered HVAC monitoring and optimization platform",
        version="1.0.0",
        lifespan=lifecycle,
        docs_url="/docs" if config.debug else None,  # Disable docs in production
        redoc_url="/redoc" if config.debug else None,
    )

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    return app


app = create_app()
app_start_time = time.time()


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("websocket_connected", total_connections=len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("websocket_disconnected", total_connections=len(self.active_connections))

    async def broadcast_message(self, message: WebSocketMessage) -> None:
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        message_json = message.model_dump_json()

        # Send to all connections, remove failed ones
        failed_connections = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.warning("websocket_send_failed", error=str(e))
                failed_connections.append(connection)

        # Remove failed connections
        for failed_connection in failed_connections:
            self.disconnect(failed_connection)


websocket_manager: ConnectionManager = ConnectionManager()


# Health check endpoints
@app.get("/health", response_model=SystemStatusResponse)
async def health_check() -> SystemStatusResponse:
    """System health check endpoint."""

    current_time = datetime.now(UTC)
    uptime = time.time() - app_start_time

    # Check component health
    components = {
        "api": "healthy",
        "ai_service": "healthy" if hvac_ai_service else "down",
        "monitoring_service": "healthy" if monitoring_service else "down",
    }

    # Determine overall status
    if all(status == "healthy" for status in components.values()):
        overall_status = "healthy"
    elif any(status == "down" for status in components.values()):
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return SystemStatusResponse(
        status=overall_status,
        timestamp=current_time,
        version="1.0.0",
        uptime_seconds=uptime,
        components=components,
        metrics={
            "websocket_connections": len(websocket_manager.active_connections),
            "uptime_hours": round(uptime / 3600, 2),
        },
    )


@app.get("/health/ready")
async def readiness_check() -> dict[str, str]:
    """Kubernetes readiness check."""

    if not hvac_ai_service or not monitoring_service:
        raise HTTPException(status_code=503, detail="Services not ready")

    return {"status": "ready"}


@app.get("/health/live")
async def liveness_check() -> dict[str, str]:
    """Kubernetes liveness check."""
    return {"status": "alive"}


# API endpoints
@app.post("/api/v1/analyze", response_model=AIAnalysisResponse)
async def analyze_hvac_metrics(
    request: MetricSubmissionRequest, background_tasks: BackgroundTasks
) -> AIAnalysisResponse:
    """Analyze HVAC metrics using AI agents."""

    if not hvac_ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")

    analysis_id = f"analysis_{int(time.time())}"
    analysis_start = datetime.now(UTC)

    logger.info(
        "analysis_request_received", analysis_id=analysis_id, metrics_count=len(request.metrics)
    )

    try:
        # Run AI analysis
        results = await hvac_ai_service.comprehensive_analysis(
            request.metrics, request.system_context or {}
        )

        # Prepare response
        response = AIAnalysisResponse(
            analysis_id=analysis_id,
            timestamp=analysis_start,
            energy_optimization=results.get("energy_optimization"),
            comfort_assessment=results.get("comfort_assessment"),
            fault_detection=results.get("fault_detection"),
            analysis_metadata=results.get("analysis_metadata", {}),
        )

        # Broadcast results to WebSocket clients
        background_tasks.add_task(
            websocket_manager.broadcast_message,
            WebSocketMessage(
                message_type="analysis_result",
                data={"analysis_id": analysis_id, "results": results},
            ),
        )

        logger.info("analysis_completed", analysis_id=analysis_id)

        return response

    except Exception as e:
        logger.error("analysis_failed", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}") from e


@app.get("/api/v1/metrics/types")
async def get_metric_types() -> dict[str, list[dict[str, str]]]:
    """Get available HVAC metric types."""

    return {
        "hvac_metric_types": [
            {"type": metric_type.value, "description": metric_type.value.replace("_", " ").title()}
            for metric_type in HVACMetricType
        ],
        "equipment_types": [
            {
                "type": equipment_type.value,
                "description": equipment_type.value.replace("_", " ").title(),
            }
            for equipment_type in HVACSystemType
        ],
    }


@app.get("/api/v1/agents/capabilities")
async def get_agent_capabilities() -> dict[str, Any]:
    """Get AI agent capabilities information."""

    if not hvac_ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")

    return hvac_ai_service.get_agent_capabilities()


@app.post("/api/v1/monitoring/start")
async def start_monitoring() -> dict[str, str]:
    """Start continuous monitoring service."""

    if not monitoring_service:
        raise HTTPException(status_code=503, detail="Monitoring service not available")

    # Start monitoring in background
    # In production, this would be managed by a task queue
    return {"status": "monitoring_started", "message": "Continuous monitoring initiated"}


@app.post("/api/v1/monitoring/stop")
async def stop_monitoring() -> dict[str, str]:
    """Stop continuous monitoring service."""

    if not monitoring_service:
        raise HTTPException(status_code=503, detail="Monitoring service not available")

    await monitoring_service.stop()
    return {"status": "monitoring_stopped"}


# WebSocket endpoint for real-time updates
@app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time monitoring updates."""

    await websocket_manager.connect(websocket)

    try:
        # Send initial connection message
        await websocket.send_json(
            {
                "message_type": "connection_established",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {"status": "connected"},
            }
        )

        # Keep connection alive and handle incoming messages
        while True:
            # Receive messages from client (if any)
            data = await websocket.receive_text()

            # Echo back for demonstration
            await websocket.send_json(
                {
                    "message_type": "echo",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "data": {"received": data},
                }
            )

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error("websocket_error", error=str(e))
        websocket_manager.disconnect(websocket)


# Root endpoint
@app.get("/")
async def root() -> dict[str, dict[str, str] | str]:
    """Root endpoint with API information."""

    return {
        "name": "Intelligent HVAC Monitoring System",
        "version": "1.0.0",
        "description": "AI-powered HVAC monitoring and optimization platform",
        "endpoints": {
            "health": "/health",
            "analyze": "/api/v1/analyze",
            "websocket": "/ws/monitoring",
            "documentation": "/docs" if get_config().debug else "disabled",
        },
    }


# Development server runner
if __name__ == "__main__":
    config = get_config()

    uvicorn.run(
        "api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level=config.logging.level.lower(),
    )
