from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def setup_logging(name: str = "backend") -> logging.Logger:
    """Configure module-level logger with stdout handler."""
    log = logging.getLogger(name)
    if log.handlers:
        return log

    log.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    log.addHandler(console)

    return log


logger = setup_logging()


@contextmanager
def timer(label: str) -> Generator[dict[str, float], None, None]:
    """Context manager that records elapsed milliseconds."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000.0
        logger.debug("%s completed in %.1f ms", label, result["elapsed_ms"])


def timed(label: str | None = None) -> Callable[[F], F]:
    """Decorator to log function execution time."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = label or func.__name__
            with timer(name) as t:
                out = func(*args, **kwargs)
            logger.info("%s took %.1f ms", name, t["elapsed_ms"])
            return out

        return wrapper  # type: ignore[return-value]

    return decorator
