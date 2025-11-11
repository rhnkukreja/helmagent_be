# utils/memory_logger.py
import os
import tracemalloc
from datetime import datetime
from functools import wraps
from typing import Callable, Any
from fastapi import Request
import logging
import logging.handlers
import asyncio
# ----------------------------------------------------------------------
# 1. Configure a rotating file handler (max 5 MB, keep 3 backups)
# ----------------------------------------------------------------------
LOG_FILE = os.path.join(os.path.dirname(__file__), "memory_logs.txt")
handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
formatter = logging.Formatter("%(message)s")   # we build the line ourselves
handler.setFormatter(formatter)

memory_logger = logging.getLogger("memory_logger")
memory_logger.setLevel(logging.INFO)
memory_logger.addHandler(handler)
memory_logger.propagate = False   # avoid double logging to console


# ----------------------------------------------------------------------
# 2. Helper that returns a human-readable size
# ----------------------------------------------------------------------
def _sizeof_fmt(num: float, suffix: str = "B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Pi{suffix}"


# ----------------------------------------------------------------------
# 3. The decorator
# ----------------------------------------------------------------------
def log_memory_usage(func: Callable) -> Callable:
    """
    Decorator for FastAPI endpoints.
    - Starts tracemalloc before the call
    - Captures memory **before**, **after** and **peak** (max allocated)
    - Writes ONE line to memory_logs.txt:
        TIMESTAMP | METHOD PATH | before: X.XX MiB | after: Y.YY MiB | peak: Z.ZZ MiB
    """
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        # --------------------------------------------------------------
        # Find the FastAPI Request object (it is the first positional arg
        # for normal endpoints; for WebSocket it is `websocket`)
        # --------------------------------------------------------------
        request: Request | None = None
        for a in args:
            if isinstance(a, Request):
                request = a
                break

        method_path = "UNKNOWN"
        if request:
            method_path = f"{request.method} {request.url.path}"

        # --------------------------------------------------------------
        # Start tracing
        # --------------------------------------------------------------
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        before_size = sum(stat.size for stat in before.statistics("lineno"))

        try:
            result = await func(*args, **kwargs)
        finally:
            # --------------------------------------------------------------
            # Capture after & peak
            # --------------------------------------------------------------
            after = tracemalloc.take_snapshot()
            after_size = sum(stat.size for stat in after.statistics("lineno"))
            peak = tracemalloc.get_traced_memory()[1]   # (current, peak)

            tracemalloc.stop()

            # --------------------------------------------------------------
            # Build log line
            # --------------------------------------------------------------
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            line = (
                f"{ts} | {method_path} | "
                f"before: {_sizeof_fmt(before_size)} | "
                f"after: {_sizeof_fmt(after_size)} | "
                f"peak: {_sizeof_fmt(peak)}"
            )
            memory_logger.info(line)

        return result

    # ------------------------------------------------------------------
    # Synchronous endpoints (rare, but we support them)
    # ------------------------------------------------------------------
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        request: Request | None = None
        for a in args:
            if isinstance(a, Request):
                request = a
                break
        method_path = f"{request.method} {request.url.path}" if request else "UNKNOWN"

        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        before_size = sum(stat.size for stat in before.statistics("lineno"))

        try:
            result = func(*args, **kwargs)
        finally:
            after = tracemalloc.take_snapshot()
            after_size = sum(stat.size for stat in after.statistics("lineno"))
            peak = tracemalloc.get_traced_memory()[1]

            tracemalloc.stop()

            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            line = (
                f"{ts} | {method_path} | "
                f"before: {_sizeof_fmt(before_size)} | "
                f"after: {_sizeof_fmt(after_size)} | "
                f"peak: {_sizeof_fmt(peak)}"
            )
            memory_logger.info(line)

        return result

    # Return the correct wrapper depending on whether the original is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper