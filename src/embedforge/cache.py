"""SQLite embedding cache keyed by H(provider, model, dimensions, input)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embedforge.errors import EmbedForgeError
from embedforge.paths import embeddings_cache_dir
from embedforge.shapes import utc_now

_DATABASE_NAME = "embeddings.sqlite3"


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
    created_at: str


@dataclass(frozen=True)
class CacheMigration:
    scanned: int
    imported: int
    existing: int
    deleted: int


@dataclass(frozen=True)
class _LegacyEntry:
    metadata: CacheEntry
    embedding: list[float]


class EmbeddingCache:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or embeddings_cache_dir()
        self._connection: sqlite3.Connection | None = None

    @property
    def database_path(self) -> Path:
        return self.root / _DATABASE_NAME

    def path_for(self, key: str) -> Path:
        """Return the path used by the legacy JSON cache."""
        return self.root / key[:2] / f"{key}.json"

    def get(self, provider: str, model: str, dimensions: int, text: str) -> list[float] | None:
        key = cache_key(provider, model, dimensions, text)
        cached = self.get_by_key(key, dimensions=dimensions)
        if cached is not None:
            return cached
        legacy = _load_legacy(self.path_for(key), strict=False)
        if legacy is None:
            return None
        metadata = legacy.metadata
        if (
            metadata.provider != provider
            or metadata.model != model
            or metadata.dimensions != dimensions
        ):
            return None
        self.put_by_key(
            key,
            provider=provider,
            model=model,
            dimensions=dimensions,
            embedding=legacy.embedding,
            created_at=metadata.created_at,
        )
        return self.get_by_key(key, dimensions=dimensions)

    def get_by_key(self, key: str, *, dimensions: int | None = None) -> list[float] | None:
        if not self.database_path.exists() and self._connection is None:
            return None
        row = self._connect().execute(
            "SELECT dimensions, embedding FROM embeddings WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        stored_dimensions = int(row[0])
        if dimensions is not None and stored_dimensions != dimensions:
            raise EmbedForgeError(
                f"cache entry {key} has {stored_dimensions} dimensions; expected {dimensions}"
            )
        return _decode_embedding(bytes(row[1]), stored_dimensions)

    def put(
        self,
        provider: str,
        model: str,
        dimensions: int,
        text: str,
        embedding: list[float],
    ) -> str:
        key = cache_key(provider, model, dimensions, text)
        self.put_by_key(
            key,
            provider=provider,
            model=model,
            dimensions=dimensions,
            embedding=embedding,
        )
        return key

    def matches(self, key: str, *, dimensions: int, embedding: list[float]) -> bool:
        cached = self.get_by_key(key, dimensions=dimensions)
        if cached is None:
            return False
        return _encode_embedding(cached, dimensions) == _encode_embedding(
            embedding,
            dimensions,
        )

    def put_by_key(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        dimensions: int,
        embedding: list[float],
        created_at: str | None = None,
    ) -> None:
        payload = _encode_embedding(embedding, dimensions)
        connection = self._connect()
        with connection:
            connection.execute(
                """
                INSERT INTO embeddings(key, provider, model, dimensions, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    provider = excluded.provider,
                    model = excluded.model,
                    dimensions = excluded.dimensions,
                    embedding = excluded.embedding,
                    created_at = excluded.created_at
                """,
                (
                    key,
                    provider,
                    model,
                    dimensions,
                    sqlite3.Binary(payload),
                    created_at or utc_now(),
                ),
            )

    def list_entries(self) -> list[CacheEntry]:
        entries: dict[str, CacheEntry] = {}
        if self.database_path.exists() or self._connection is not None:
            rows = self._connect().execute(
                "SELECT key, provider, model, dimensions, created_at FROM embeddings ORDER BY key"
            )
            for key, provider, model, dimensions, created_at in rows:
                entries[str(key)] = CacheEntry(
                    key=str(key),
                    provider=str(provider),
                    model=str(model),
                    dimensions=int(dimensions),
                    created_at=str(created_at),
                )
        for path in self._legacy_paths():
            key = path.stem
            if key in entries:
                continue
            legacy = _load_legacy(path, strict=False)
            if legacy is not None:
                entries[legacy.metadata.key] = legacy.metadata
        return [entries[key] for key in sorted(entries)]

    def migrate_legacy(self, *, delete: bool = False) -> CacheMigration:
        paths = self._legacy_paths()
        imported = 0
        existing = 0
        connection = self._connect()
        with connection:
            for path in paths:
                legacy = _load_legacy(path, strict=True)
                assert legacy is not None
                entry = legacy.metadata
                payload = _encode_embedding(legacy.embedding, entry.dimensions)
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO embeddings(
                        key, provider, model, dimensions, embedding, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.key,
                        entry.provider,
                        entry.model,
                        entry.dimensions,
                        sqlite3.Binary(payload),
                        entry.created_at,
                    ),
                )
                if cursor.rowcount == 1:
                    imported += 1
                else:
                    stored = connection.execute(
                        """
                        SELECT provider, model, dimensions, embedding, created_at
                        FROM embeddings WHERE key = ?
                        """,
                        (entry.key,),
                    ).fetchone()
                    if stored is None or (
                        str(stored[0]) != entry.provider
                        or str(stored[1]) != entry.model
                        or int(stored[2]) != entry.dimensions
                        or bytes(stored[3]) != payload
                        or str(stored[4]) != entry.created_at
                    ):
                        raise EmbedForgeError(
                            f"legacy cache entry {path} does not match its SQLite copy"
                        )
                    existing += 1
        deleted = 0
        if delete:
            for path in paths:
                path.unlink()
                deleted += 1
            self._remove_empty_legacy_directories()
        return CacheMigration(
            scanned=len(paths),
            imported=imported,
            existing=existing,
            deleted=deleted,
        )

    def info(self) -> dict[str, Any]:
        size = 0
        if self.root.exists():
            for path in self.root.rglob("*"):
                if path.is_file():
                    size += path.stat().st_size
        return {
            "path": str(self.root),
            "database": str(self.database_path),
            "entries": len(self.list_entries()),
            "bytes": size,
            "legacy_entries": len(self._legacy_paths()),
        }

    def clean(self) -> int:
        entries = self.list_entries()
        self.close()
        if self.root.exists():
            for path in self.root.rglob("*"):
                if path.is_file():
                    path.unlink()
            for directory in sorted(self.root.rglob("*"), reverse=True):
                if directory.is_dir():
                    directory.rmdir()
        return len(entries)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings(
                key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL CHECK(dimensions > 0),
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()
        self._connection = connection
        return connection

    def _legacy_paths(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("*/*.json"))

    def _remove_empty_legacy_directories(self) -> None:
        for directory in sorted(self.root.glob("*"), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()


def _encode_embedding(embedding: list[float], dimensions: int) -> bytes:
    if len(embedding) != dimensions:
        raise EmbedForgeError(
            f"embedding has {len(embedding)} dimensions; expected {dimensions}"
        )
    try:
        values = array("f", (float(value) for value in embedding))
    except (OverflowError, TypeError, ValueError) as exc:
        raise EmbedForgeError(f"embedding cannot be stored as float32: {exc}") from exc
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _decode_embedding(payload: bytes, dimensions: int) -> list[float]:
    expected = dimensions * 4
    if len(payload) != expected:
        raise EmbedForgeError(
            f"cache embedding contains {len(payload)} bytes; expected {expected}"
        )
    values = array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def _load_legacy(path: Path, *, strict: bool) -> _LegacyEntry | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("expected a JSON object")
        embedding = raw.get("embedding")
        key = raw.get("key")
        provider = raw.get("provider")
        model = raw.get("model")
        dimensions = raw.get("dimensions")
        created_at = raw.get("created_at")
        if not (
            isinstance(embedding, list)
            and isinstance(key, str)
            and isinstance(provider, str)
            and isinstance(model, str)
            and isinstance(dimensions, int)
            and not isinstance(dimensions, bool)
            and isinstance(created_at, str)
        ):
            raise ValueError("missing or invalid fields")
        if path.stem != key:
            raise ValueError("file name does not match cache key")
        vector = [float(value) for value in embedding]
        if len(vector) != dimensions:
            raise ValueError(
                f"embedding has {len(vector)} dimensions; expected {dimensions}"
            )
        return _LegacyEntry(
            metadata=CacheEntry(
                key=key,
                provider=provider,
                model=model,
                dimensions=dimensions,
                created_at=created_at,
            ),
            embedding=vector,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if strict:
            raise EmbedForgeError(f"cannot migrate legacy cache entry {path}: {exc}") from exc
        return None
