from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_hash(file_path: Path) -> str:
    """Compute 16-character SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def compute_string_hash(text: str) -> str:
    """Compute 16-character SHA-256 hash of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
