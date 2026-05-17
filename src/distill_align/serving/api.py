"""FastAPI application for serving fine-tuned language models (OpenAI-compatible)."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# SECTION 1: Pydantic Request/Response Models
# ═══════════════════════════════════════════════


class GenerateRequest(BaseModel):
    """Request body for the /v1/generate endpoint."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32768,
        description="Input text to generate from. Must not be empty.",
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = greedy, higher = more random.",
    )
    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling: only consider tokens with cumulative prob > top_p.",
    )
    top_k: int = Field(
        default=0,
        ge=0,
        description="Top-k sampling. 0 = disabled (use top_p instead).",
    )
    stop: list[str] = Field(
        default_factory=list,
        description="Stop sequences. Generation halts when any of these are produced.",
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Penalty for repeating tokens. 1.0 = no penalty.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response token by token.",
    )


class GenerateResponse(BaseModel):
    """Response body for the /v1/generate endpoint."""

    text: str = Field(description="Generated text (excluding the prompt).")
    finish_reason: str = Field(
        description="Why generation stopped: 'stop', 'length', or 'error'."
    )
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage: prompt_tokens, completion_tokens, total_tokens.",
    )
    latency_ms: float = Field(description="Generation time in milliseconds.")
    model: str = Field(description="Model that produced this response.")


class ModelInfoResponse(BaseModel):
    """Response body for the /v1/model/info endpoint."""

    model_name: str = Field(description="Name or path of the loaded model.")
    is_ready: bool = Field(description="Whether the model is ready to serve.")
    engine: str = Field(description="Backend engine: 'vllm' or 'huggingface'.")
    max_model_len: int = Field(description="Maximum supported sequence length.")


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str = Field(description="Service status: 'healthy' or 'unhealthy'.")
    model_loaded: bool = Field(description="Whether a model is currently loaded.")
    uptime_seconds: float = Field(description="Time since service started.")


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(description="Error type.")
    message: str = Field(description="Human-readable error description.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error context.",
    )


# ═══════════════════════════════════════════════
# SECTION 2: FastAPI Application
# ═══════════════════════════════════════════════


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.

    Usage:
        app = create_app()
        app.state.engine = ServingEngine()
        app.state.engine.load_model("./model", config)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    app = FastAPI(
        title="Distill + Align LLM Serving API",
        description=(
            "REST API for serving fine-tuned language models. "
            "Compatible with OpenAI API format."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.state.start_time = time.time()
    app.state.engine = None
    app.state.monitoring = None

    # ─── Route: POST /v1/generate ─────────────────────────────────

    @app.post(
        "/v1/generate",
        response_model=GenerateResponse,
        responses={
            422: {"model": ErrorResponse, "description": "Validation error"},
            503: {"model": ErrorResponse, "description": "Service unavailable"},
        },
        summary="Generate text from a prompt",
        description="Generate text using the loaded model. Returns the full response at once.",
    )
    async def generate(request: GenerateRequest) -> GenerateResponse:
        """Handle synchronous text generation requests."""
        engine = app.state.engine
        if engine is None or not engine.is_ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_unavailable",
                    "message": "Model is not loaded. The service is starting up.",
                },
            )

        from distill_align.serving.engine import GenerationRequest

        engine_request = GenerationRequest(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            repetition_penalty=request.repetition_penalty,
        )

        try:
            result = engine.generate(engine_request)

            if app.state.monitoring:
                app.state.monitoring.record_request(
                    latency_ms=result.latency_ms,
                    input_tokens=result.usage.get("prompt_tokens", 0),
                    output_tokens=result.usage.get("completion_tokens", 0),
                    status="success",
                )

            return GenerateResponse(
                text=result.text,
                finish_reason=result.finish_reason,
                usage=result.usage,
                latency_ms=result.latency_ms,
                model=result.model_name,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if app.state.monitoring:
                app.state.monitoring.record_request(
                    latency_ms=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    status="error",
                )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "generation_failed",
                    "message": str(e),
                },
            ) from e

    # ─── Route: POST /v1/generate/stream ──────────────────────────

    @app.post(
        "/v1/generate/stream",
        summary="Generate text with streaming (SSE)",
        description=(
            "Generate text using Server-Sent Events. "
            "Each event contains a text chunk as it's generated."
        ),
        responses={
            503: {"model": ErrorResponse, "description": "Service unavailable"},
        },
    )
    async def generate_stream(request: GenerateRequest) -> StreamingResponse:
        """Handle streaming text generation via Server-Sent Events (SSE)."""
        engine = app.state.engine
        if engine is None or not engine.is_ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_unavailable",
                    "message": "Model is not loaded. The service is starting up.",
                },
            )

        from distill_align.serving.engine import GenerationRequest

        engine_request = GenerationRequest(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            stream=True,
            repetition_penalty=request.repetition_penalty,
        )

        async def event_generator():
            """Async generator that yields SSE-formatted events."""
            import json

            start_time = time.perf_counter()
            total_tokens = 0

            try:
                async for chunk in engine.generate_stream(engine_request):
                    total_tokens += 1
                    event_data = json.dumps({
                        "text": chunk.text,
                        "finish_reason": chunk.finish_reason,
                    })
                    yield f"data: {event_data}\n\n"

                yield "data: [DONE]\n\n"

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                if app.state.monitoring:
                    app.state.monitoring.record_request(
                        latency_ms=latency_ms,
                        input_tokens=0,
                        output_tokens=total_tokens,
                        status="success",
                    )

            except Exception as e:
                logger.error(f"Streaming generation failed: {e}")
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ─── Route: GET /v1/model/info ────────────────────────────────

    @app.get(
        "/v1/model/info",
        response_model=ModelInfoResponse,
        summary="Get model information",
        description="Returns metadata about the currently loaded model.",
    )
    async def model_info() -> ModelInfoResponse:
        """Return information about the loaded model."""
        engine = app.state.engine

        if engine is None:
            return ModelInfoResponse(
                model_name="none",
                is_ready=False,
                engine="none",
                max_model_len=0,
            )

        config = engine._config
        return ModelInfoResponse(
            model_name=engine.model_name,
            is_ready=engine.is_ready,
            engine="vllm" if engine._using_vllm else "huggingface",
            max_model_len=getattr(config, "max_model_len", 2048) if config else 2048,
        )

    # ─── Route: GET /health ───────────────────────────────────────

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check",
        description="Returns service health status. Used by load balancers and orchestrators.",
    )
    async def health_check() -> HealthResponse:
        """Health check endpoint for load balancers and container orchestrators."""
        engine = app.state.engine
        model_loaded = engine is not None and engine.is_ready

        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            uptime_seconds=time.time() - app.state.start_time,
        )

    # ─── Route: GET /metrics ──────────────────────────────────────

    @app.get(
        "/metrics",
        summary="Prometheus metrics",
        description="Returns metrics in Prometheus exposition format for scraping.",
        response_class=StreamingResponse,
    )
    async def prometheus_metrics(request: Request) -> StreamingResponse:
        """Expose Prometheus metrics for monitoring."""
        monitoring = app.state.monitoring

        if monitoring is not None:
            metrics_text = monitoring.get_prometheus_metrics()
        else:
            uptime = time.time() - app.state.start_time
            engine = app.state.engine
            model_loaded = 1 if (engine and engine.is_ready) else 0
            metrics_text = (
                "# HELP distill_align_up Whether the service is up\n"
                "# TYPE distill_align_up gauge\n"
                f"distill_align_up {model_loaded}\n"
                "# HELP distill_align_uptime_seconds Service uptime\n"
                "# TYPE distill_align_uptime_seconds gauge\n"
                f"distill_align_uptime_seconds {uptime:.1f}\n"
            )

        return StreamingResponse(
            iter([metrics_text]),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


# ═══════════════════════════════════════════════
# SECTION 3: Application Instance
# ═══════════════════════════════════════════════

app = create_app()
