"""Monitoring Service using Prometheus metrics for the serving layer."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# SECTION 1: Alert Dataclass
# ═══════════════════════════════════════════════


@dataclass
class Alert:
    """Represents an active monitoring alert.

    Attributes:
        name: Alert identifier (e.g., "high_error_rate", "high_p99_latency")
        severity: Alert severity level ("warning", "critical")
        message: Human-readable description of the alert condition
        current_value: The metric value that triggered the alert
        threshold: The configured threshold that was exceeded
        timestamp: When the alert was triggered (Unix timestamp)
    """

    name: str
    severity: str
    message: str
    current_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════
# SECTION 2: MonitoringService Class
# ═══════════════════════════════════════════════


class MonitoringService:
    """Production monitoring service using Prometheus metrics.

    Records metrics for every request, maintains a rolling window for alert
    evaluation, exposes metrics in Prometheus format, and checks alert conditions.

    Usage:
        monitoring = MonitoringService(
            error_rate_threshold=0.05,
            p99_latency_threshold_ms=5000,
        )

        monitoring.record_request(
            latency_ms=120.5,
            input_tokens=50,
            output_tokens=100,
            status="success"
        )

        alerts = monitoring.check_alerts()
        metrics_text = monitoring.get_prometheus_metrics()
    """

    def __init__(
        self,
        error_rate_threshold: float = 0.05,
        p99_latency_threshold_ms: float = 5000.0,
        rolling_window_seconds: int = 300,
    ) -> None:
        """Initialize the monitoring service.

        Args:
            error_rate_threshold: Alert if error rate exceeds this fraction.
            p99_latency_threshold_ms: Alert if p99 latency exceeds this (ms).
            rolling_window_seconds: Size of the rolling window for alert evaluation.
        """
        self._error_rate_threshold = error_rate_threshold
        self._p99_latency_threshold_ms = p99_latency_threshold_ms
        self._rolling_window_seconds = rolling_window_seconds

        self._request_history: deque[dict[str, Any]] = deque()

        self._total_requests: int = 0
        self._total_errors: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_latency_ms: float = 0.0

        self._prometheus_initialized: bool = False
        self._prom_metrics: dict[str, Any] = {}
        self._init_prometheus_metrics()

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        status: str,
    ) -> None:
        """Record metrics for a single request.

        Args:
            latency_ms: Total request latency in milliseconds.
            input_tokens: Number of tokens in the prompt.
            output_tokens: Number of tokens generated.
            status: Request outcome — "success" or "error".
        """
        now = time.time()

        self._request_history.append({
            "timestamp": now,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "status": status,
        })

        self._total_requests += 1
        self._total_latency_ms += latency_ms
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        if status == "error":
            self._total_errors += 1

        if self._prometheus_initialized:
            self._prom_metrics["request_count"].labels(status=status).inc()
            self._prom_metrics["request_latency"].observe(latency_ms)
            self._prom_metrics["input_tokens"].inc(input_tokens)
            self._prom_metrics["output_tokens"].inc(output_tokens)

        self._prune_rolling_window(now)

    def check_alerts(self) -> list[Alert]:
        """Evaluate alert conditions against current metrics.

        Returns:
            List of active Alert objects. Empty list = all clear.
        """
        alerts: list[Alert] = []
        now = time.time()

        self._prune_rolling_window(now)

        if len(self._request_history) < 5:
            return alerts

        # ─── Check 1: Error Rate ─────────────────────────────────
        recent_requests = list(self._request_history)
        total_recent = len(recent_requests)
        error_count = sum(1 for r in recent_requests if r["status"] == "error")
        error_rate = error_count / total_recent if total_recent > 0 else 0.0

        if error_rate > self._error_rate_threshold:
            alerts.append(Alert(
                name="high_error_rate",
                severity="critical" if error_rate > self._error_rate_threshold * 2 else "warning",
                message=(
                    f"Error rate {error_rate:.1%} exceeds threshold "
                    f"{self._error_rate_threshold:.1%} "
                    f"({error_count}/{total_recent} requests failed in last "
                    f"{self._rolling_window_seconds}s)"
                ),
                current_value=error_rate,
                threshold=self._error_rate_threshold,
            ))

        # ─── Check 2: P99 Latency ────────────────────────────────
        latencies = sorted(r["latency_ms"] for r in recent_requests if r["status"] == "success")
        if latencies:
            p99_index = int(len(latencies) * 0.99)
            p99_latency = latencies[min(p99_index, len(latencies) - 1)]

            if p99_latency > self._p99_latency_threshold_ms:
                alerts.append(Alert(
                    name="high_p99_latency",
                    severity="critical" if p99_latency > self._p99_latency_threshold_ms * 2 else "warning",
                    message=(
                        f"P99 latency {p99_latency:.0f}ms exceeds threshold "
                        f"{self._p99_latency_threshold_ms:.0f}ms "
                        f"(based on {len(latencies)} successful requests)"
                    ),
                    current_value=p99_latency,
                    threshold=self._p99_latency_threshold_ms,
                ))

        return alerts

    def get_prometheus_metrics(self) -> str:
        """Generate metrics in Prometheus text exposition format.

        Returns:
            String in Prometheus exposition format.
        """
        if self._prometheus_initialized:
            try:
                from prometheus_client import generate_latest

                return generate_latest().decode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to generate Prometheus metrics: {e}")

        return self._generate_metrics_manually()

    # ─── Private helper methods ───────────────────────────────────

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metric objects (falls back to manual generation)."""
        try:
            from prometheus_client import CollectorRegistry, Counter, Histogram

            self._registry = CollectorRegistry()

            self._prom_metrics["request_count"] = Counter(
                "distill_align_requests_total",
                "Total number of generation requests",
                labelnames=["status"],
                registry=self._registry,
            )

            self._prom_metrics["request_latency"] = Histogram(
                "distill_align_request_latency_ms",
                "Request latency in milliseconds",
                buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                registry=self._registry,
            )

            self._prom_metrics["input_tokens"] = Counter(
                "distill_align_input_tokens_total",
                "Total input tokens processed",
                registry=self._registry,
            )
            self._prom_metrics["output_tokens"] = Counter(
                "distill_align_output_tokens_total",
                "Total output tokens generated",
                registry=self._registry,
            )

            self._prometheus_initialized = True
            logger.debug("Prometheus metrics initialized.")

        except ImportError:
            logger.info(
                "prometheus_client not installed. Using manual metric generation. "
                "Install with: pip install prometheus-client"
            )
            self._prometheus_initialized = False

    def _generate_metrics_manually(self) -> str:
        """Generate Prometheus-format metrics without prometheus_client.

        Returns:
            Prometheus exposition format string.
        """
        recent = list(self._request_history)
        error_count = sum(1 for r in recent if r["status"] == "error")

        latencies = sorted(r["latency_ms"] for r in recent if r["status"] == "success")
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

        lines = [
            "# HELP distill_align_requests_total Total number of generation requests",
            "# TYPE distill_align_requests_total counter",
            f'distill_align_requests_total{{status="success"}} {self._total_requests - self._total_errors}',
            f'distill_align_requests_total{{status="error"}} {self._total_errors}',
            "",
            "# HELP distill_align_request_latency_ms Request latency in milliseconds",
            "# TYPE distill_align_request_latency_ms summary",
            f'distill_align_request_latency_ms{{quantile="0.5"}} {p50:.1f}',
            f'distill_align_request_latency_ms{{quantile="0.99"}} {p99:.1f}',
            f"distill_align_request_latency_ms_sum {self._total_latency_ms:.1f}",
            f"distill_align_request_latency_ms_count {self._total_requests}",
            "",
            "# HELP distill_align_input_tokens_total Total input tokens processed",
            "# TYPE distill_align_input_tokens_total counter",
            f"distill_align_input_tokens_total {self._total_input_tokens}",
            "",
            "# HELP distill_align_output_tokens_total Total output tokens generated",
            "# TYPE distill_align_output_tokens_total counter",
            f"distill_align_output_tokens_total {self._total_output_tokens}",
            "",
            "# HELP distill_align_error_rate Current error rate (rolling window)",
            "# TYPE distill_align_error_rate gauge",
            f"distill_align_error_rate {error_count / max(len(recent), 1):.4f}",
            "",
        ]

        return "\n".join(lines) + "\n"

    def _prune_rolling_window(self, now: float) -> None:
        """Remove entries older than the rolling window from history.

        Args:
            now: Current Unix timestamp.
        """
        cutoff = now - self._rolling_window_seconds
        while self._request_history and self._request_history[0]["timestamp"] < cutoff:
            self._request_history.popleft()
