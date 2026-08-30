"""Backend utility exports without import-time service connections.

Historically this package imported ``redis_cache`` eagerly, so importing an
unrelated helper such as ``backend.utils.logging`` could block on a Redis ping.
Exports remain compatible but are resolved only when explicitly requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "compute_file_hash": "backend.utils.hashing",
    "compute_string_hash": "backend.utils.hashing",
    "logger": "backend.utils.logging",
    "setup_logging": "backend.utils.logging",
    "timed": "backend.utils.logging",
    "timer": "backend.utils.logging",
    "RedisCache": "backend.utils.redis_cache",
    "get_redis_cache": "backend.utils.redis_cache",
    "redis_cache": "backend.utils.redis_cache",
    "SECTION_PATTERNS": "backend.utils.section_tracker",
    "SectionContext": "backend.utils.section_tracker",
    "SectionHeading": "backend.utils.section_tracker",
    "SectionPattern": "backend.utils.section_tracker",
    "SectionTracker": "backend.utils.section_tracker",
    "clean_title": "backend.utils.section_tracker",
    "is_noise_line": "backend.utils.section_tracker",
    "parse_section_heading": "backend.utils.section_tracker",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))


__all__ = sorted(_EXPORT_MODULES)
