"""Embedding cache keyed by H(provider, model, dimensions, input)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embedforge.paths import embeddings_cache_dir
from embedforge.shapes import utc_now


def cache_key(provider: str, model: str, dimensions: int, text: str) -> str:
    payload = json.dumps(
        [provider, model, dimensions, text],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    key: str
    provider: str
    model: str
    dimensions: int
    embedding: tuple[float, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "embedding": list(self.embedding),
            "created_at": self.created_at,
        }


class EmbeddingCache:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or embeddings_cache_dir()

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, provider: str, model: str, dimensions: int, text: str) -> list[float] | None:
        key = cache_key(provider, model, dimensions, text)
        path = self.path_for(key)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            return None
        embedding = raw.get("embedding")
        if not isinstance(embedding, list):
            return None
        return [float(item) for item in embedding]

    def put(
        self,
        provider: str,
        model: str,
        dimensions: int,
        text: str,
        embedding: list[float],
    ) -> str:
        key = cache_key(provider, model, dimensions, text)
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = CacheEntry(
            key=key,
            provider=provider,
            model=model,
            dimensions=dimensions,
            embedding=tuple(float(item) for item in embedding),
            created_at=utc_now(),
        )
        path.write_text(json.dumps(entry.to_dict(), indent=2) + "\n")
        return key

    def list_entries(self) -> list[CacheEntry]:
        if not self.root.exists():
            return []
        entries: list[CacheEntry] = []
        for path in sorted(self.root.glob("*/*.json")):
            raw = json.loads(path.read_text())
            if not isinstance(raw, dict):
                continue
            embedding = raw.get("embedding")
            if not isinstance(embedding, list):
                continue
            key = raw.get("key")
            provider = raw.get("provider")
            model = raw.get("model")
            dimensions = raw.get("dimensions")
            created_at = raw.get("created_at")
            if not (
                isinstance(key, str)
                and isinstance(provider, str)
                and isinstance(model, str)
                and isinstance(dimensions, int)
                and not isinstance(dimensions, bool)
                and isinstance(created_at, str)
            ):
                continue
            entries.append(
                CacheEntry(
                    key=key,
                    provider=provider,
                    model=model,
                    dimensions=dimensions,
                    embedding=tuple(float(item) for item in embedding),
                    created_at=created_at,
                )
            )
        return entries

    def info(self) -> dict[str, Any]:
        entries = self.list_entries()
        size = 0
        if self.root.exists():
            for path in self.root.rglob("*"):
                if path.is_file():
                    size += path.stat().st_size
        return {
            "path": str(self.root),
            "entries": len(entries),
            "bytes": size,
        }

    def clean(self) -> int:
        entries = self.list_entries()
        if self.root.exists():
            for path in self.root.rglob("*"):
                if path.is_file():
                    path.unlink()
            for directory in sorted(self.root.rglob("*"), reverse=True):
                if directory.is_dir():
                    directory.rmdir()
        return len(entries)
