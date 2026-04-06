"""
Structured JSON logging configuration via structlog.

Why structlog over the stdlib logging module?
- Every log call produces a JSON object, not a formatted string — this makes
  logs machine-parseable by log aggregators (Datadog, CloudWatch, Loki) without
  a regex parsing step.
- Context variables (image_id, card_type, request_id) can be bound once per
  request and automatically included in every subsequent log call within that
  scope, eliminating repetitive kwarg passing.
- The processor chain is composable — adding OpenTelemetry trace IDs later
  is a one-line addition to the chain.

Technical Debt: No distributed trace ID injection yet. When moving to
production, add an OpenTelemetry processor here to stamp every log entry
with trace_id and span_id from the active span context.
"""

import logging
import sys
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with a JSON renderer for production use.

    Call this once at application startup (FastAPI lifespan or CLI entry point)
    before any log calls are made. Subsequent calls are idempotent.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
                   Passed in from PregraderSettings so the level is
                   environment-configurable without code changes.
    """
    # Wire structlog into the stdlib logging system so third-party libraries
    # (uvicorn, tensorflow) that use stdlib logging also emit structured JSON.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            # Inject log level as a string field ("info", "error", etc.)
            structlog.stdlib.add_log_level,
            # Inject ISO-8601 timestamp — required for log aggregator time indexing.
            structlog.processors.TimeStamper(fmt="iso"),
            # Render nested dicts/exceptions as JSON-safe structures.
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Final renderer: emit a single JSON line per log call.
            structlog.processors.JSONRenderer(),
        ],
        # Use stdlib's LogRecord as the underlying event dict carrier so
        # structlog and stdlib loggers share the same level filtering.
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Cache the logger on first bind — avoids repeated processor chain
        # construction on hot paths (e.g., per-image inference logging).
        cache_logger_on_first_use=True,
    )


def get_logger(*args: Any, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, optionally pre-populated with context.

    Usage:
        logger = get_logger(service="ingestion")
        logger.info("image_validated", image_id="abc123", format="jpeg")

    Args:
        *args: Passed to structlog.get_logger (typically the module name).
        **initial_values: Key-value pairs bound to every log call on this logger.

    Returns:
        A BoundLogger instance ready for structured logging.
    """
    return structlog.get_logger(*args, **initial_values)
