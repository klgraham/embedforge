"""Config and cache locations. Jobs live under ~/.cache/embedforge."""

from __future__ import annotations

import os
from pathlib import Path


def config_dir() -> Path:
    override = os.environ.get("EMBEDFORGE_CONFIG_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "embedforge"
    return Path.home() / ".config" / "embedforge"


def config_path() -> Path:
    return config_dir() / "config.toml"


def cache_dir() -> Path:
    override = os.environ.get("EMBEDFORGE_CACHE_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "embedforge"
    return Path.home() / ".cache" / "embedforge"


def jobs_dir() -> Path:
    return cache_dir() / "jobs"


def job_dir(job_id: str) -> Path:
    return jobs_dir() / job_id


def embeddings_cache_dir() -> Path:
    return cache_dir() / "embeddings"
