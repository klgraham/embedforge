"""Verify packed cache and dataset storage with deterministic vectors."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value

from embedforge.cache import EmbeddingCache, cache_key


def _vectors(rows: int, dimensions: int) -> list[list[float]]:
    return [
        [((row * dimensions + column) % 997 - 498) / 997 for column in range(dimensions)]
        for row in range(rows)
    ]


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    rows = 200
    dimensions = 1536
    vectors = _vectors(rows, dimensions)
    with tempfile.TemporaryDirectory(prefix="embedforge-storage-") as raw_tmp:
        root = Path(raw_tmp)
        cache = EmbeddingCache(root / "cache")
        for index, vector in enumerate(vectors):
            text = f"row-{index}"
            key = cache_key("openai", "text-embedding-3-small", dimensions, text)
            path = cache.path_for(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "key": key,
                        "provider": "openai",
                        "model": "text-embedding-3-small",
                        "dimensions": dimensions,
                        "embedding": vector,
                        "created_at": "2026-09-14T00:00:00Z",
                    },
                    indent=2,
                )
                + "\n"
            )
        legacy_bytes = _size(cache.root)
        migration = cache.migrate_legacy(delete=True)
        cache.close()
        packed_bytes = _size(cache.root)

        parquet_sizes: dict[str, int] = {}
        for dtype in ("float32", "float16"):
            dataset = Dataset.from_dict(
                {"embedding": vectors},
                features=Features(
                    {"embedding": Sequence(Value(dtype), length=dimensions)}
                ),
            )
            path = root / f"{dtype}.parquet"
            dataset.to_parquet(path)
            parquet_sizes[dtype] = path.stat().st_size

        assert migration.scanned == rows
        assert migration.deleted == rows
        assert packed_bytes < legacy_bytes
        assert parquet_sizes["float16"] < parquet_sizes["float32"]
        print(f"legacy JSON cache: {legacy_bytes:,} bytes")
        print(f"packed SQLite cache: {packed_bytes:,} bytes")
        print(f"float32 Parquet: {parquet_sizes['float32']:,} bytes")
        print(f"float16 Parquet: {parquet_sizes['float16']:,} bytes")


if __name__ == "__main__":
    main()
