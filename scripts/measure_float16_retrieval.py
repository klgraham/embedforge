"""Measure nearest-neighbor stability after float16 storage quantization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--column", default="embedding")
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--neighbors", type=int, default=10)
    return parser.parse_args()


def _sample_vectors(
    loaded: Dataset | DatasetDict,
    column: str,
    limit: int,
) -> np.ndarray:
    splits = list(loaded.values()) if isinstance(loaded, DatasetDict) else [loaded]
    available = [split for split in splits if column in split.column_names and len(split) > 0]
    total_rows = sum(len(split) for split in available)
    vectors: list[list[float]] = []
    for split in available:
        count = min(len(split), max(1, round(limit * len(split) / total_rows)))
        indices = np.linspace(0, len(split) - 1, num=count, dtype=np.int64)
        for vector in split.select(indices)[column]:
            if vector is not None:
                vectors.append(vector)
    if not vectors:
        raise ValueError(f"no non-null vectors found in column {column!r}")
    return np.asarray(vectors[:limit], dtype=np.float32)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def _neighbors(matrix: np.ndarray, queries: int, count: int) -> np.ndarray:
    scores = matrix[:queries] @ matrix.T
    row_ids = np.arange(queries)
    scores[row_ids, row_ids] = -np.inf
    candidates = np.argpartition(scores, -count, axis=1)[:, -count:]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    return np.take_along_axis(candidates, order, axis=1)


def main() -> None:
    args = _parse_args()
    loaded = load_from_disk(str(args.dataset))
    baseline = _sample_vectors(loaded, args.column, args.sample_size)
    queries = min(args.queries, len(baseline))
    neighbors = min(args.neighbors, len(baseline) - 1)
    if queries < 1 or neighbors < 1:
        raise ValueError("the sample must contain at least two non-null vectors")
    quantized = baseline.astype(np.float16).astype(np.float32)
    baseline_normalized = _normalize(baseline)
    quantized_normalized = _normalize(quantized)
    retained_cosine = np.sum(baseline_normalized * quantized_normalized, axis=1)
    baseline_neighbors = _neighbors(baseline_normalized, queries, neighbors)
    quantized_neighbors = _neighbors(quantized_normalized, queries, neighbors)
    overlap = [
        len(set(before) & set(after)) / neighbors
        for before, after in zip(baseline_neighbors, quantized_neighbors, strict=True)
    ]
    top_one = np.mean(baseline_neighbors[:, 0] == quantized_neighbors[:, 0])
    print(f"vectors: {len(baseline):,}")
    print(f"dimensions: {baseline.shape[1]:,}")
    print(f"queries: {queries:,}")
    print(f"mean self-cosine: {float(np.mean(retained_cosine)):.9f}")
    print(f"minimum self-cosine: {float(np.min(retained_cosine)):.9f}")
    print(f"top-1 agreement: {float(top_one):.3%}")
    print(f"top-{neighbors} overlap: {float(np.mean(overlap)):.3%}")


if __name__ == "__main__":
    main()
