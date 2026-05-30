from __future__ import annotations

"""Structured (JSON) logging with one line per request.

Each line carries a request_id and the OpenTelemetry trace_id, so logs correlate
directly with traces (jump from a log line to the span in Jaeger) and with the
Prometheus metrics — completing the logs/metrics/traces triad. Stdlib only.
"""

import datetime
import json
import logging
import sys

LOGGER_NAME = "gateway"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.datetime.fromtimestamp(
                record.created, datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            out.update(fields)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    """Emit a structured log line; None-valued fields are dropped."""
    clean = {k: v for k, v in fields.items() if v is not None}
    logger.info(msg, extra={"fields": clean})
