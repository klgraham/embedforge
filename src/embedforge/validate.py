"""Validate a staged embedding job against source and provenance rules."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from datasets import Dataset, load_from_disk

from embedforge.errors import EmbedForgeError
from embedforge.hfdata import DatasetRequest, DatasetSource, HuggingFaceDatasetSource
from embedforge.shapes import (
    Check,
    Diagnostic,
    Job,
    JobStatus,
    Provenance,
    ValidationResult,
    replace_job,
)
from embedforge.store import JobStore


def validate_job(
    job_id: str,
    *,
    store: JobStore | None = None,
    source: DatasetSource | None = None,
    mark_status: bool = True,
) -> ValidationResult:
    job_store = store or JobStore()
    job = job_store.load(job_id)
    provenance = job_store.load_provenance(job_id)
    output = job_store.output_dir(job_id)
    if not output.exists():
        raise EmbedForgeError(f"job {job_id} has no staged output; run it first")
    dataset = load_from_disk(str(output))
    if not isinstance(dataset, Dataset):
        raise EmbedForgeError("staged output must be a single dataset split")

    gateway = source or HuggingFaceDatasetSource()
    loaded = gateway.load_rows(
        DatasetRequest(
            repository=job.source.repository,
            config=job.source.config,
            split=job.source.split,
            revision=job.source.revision,
        ),
        limit=job.limit,
    )

    checks = [
        _check_row_counts(dataset, loaded.rows),
        _check_split(job),
        _check_source_columns(dataset, loaded.columns),
        _check_embedding_column(dataset, job.embedding.column),
        _check_dimensions(dataset, job.embedding.column, job.embedding.dimensions),
        _check_finite(dataset, job.embedding.column),
        _check_revision(job, provenance.source.revision),
        _check_model_metadata(job, provenance),
        _check_license(job, provenance.source.license),
    ]
    diagnostics = _diagnostics(dataset, job.embedding.column)
    result = ValidationResult(
        ok=all(check.passed for check in checks),
        checks=tuple(checks),
        diagnostics=tuple(diagnostics),
    )
    if mark_status and result.ok and job.status in {JobStatus.COMPLETED, JobStatus.VALIDATED}:
        job_store.save_job(replace_job(job, status=JobStatus.VALIDATED))
    return result


def _column(dataset: Dataset, name: str) -> list[Any]:
    if name not in dataset.column_names:
        return []
    return list(dataset[name])


def _check_row_counts(dataset: Dataset, source_rows: list[dict[str, object]]) -> Check:
    actual = len(dataset)
    expected = len(source_rows)
    passed = actual == expected
    return Check(
        name="source_row_counts",
        passed=passed,
        message=f"output has {actual} rows; source has {expected}",
    )


def _check_split(job: Job) -> Check:
    passed = bool(job.source.split)
    return Check(
        name="split_structure",
        passed=passed,
        message=f"source split recorded as {job.source.split!r}"
        if passed
        else "source split was not recorded",
    )


def _check_source_columns(dataset: Dataset, source_columns: tuple[str, ...]) -> Check:
    missing = [name for name in source_columns if name not in dataset.column_names]
    passed = not missing
    return Check(
        name="source_columns",
        passed=passed,
        message="all source columns present"
        if passed
        else f"missing source columns: {', '.join(missing)}",
    )


def _check_embedding_column(dataset: Dataset, column: str) -> Check:
    passed = column in dataset.column_names
    return Check(
        name="embedding_column",
        passed=passed,
        message=f"column {column!r} present" if passed else f"column {column!r} missing",
    )


def _vectors(dataset: Dataset, column: str) -> list[list[float] | None]:
    values: list[list[float] | None] = []
    for item in _column(dataset, column):
        if item is None:
            values.append(None)
        elif isinstance(item, list) and all(isinstance(num, (int, float)) for num in item):
            values.append([float(num) for num in item])
        else:
            values.append(None)
    return values


def _check_dimensions(dataset: Dataset, column: str, expected: int) -> Check:
    if column not in dataset.column_names:
        return Check(
            name="uniform_dimensions",
            passed=False,
            message="embedding column missing",
        )
    dims = {len(vector) for vector in _vectors(dataset, column) if vector is not None}
    missing = sum(1 for vector in _vectors(dataset, column) if vector is None)
    passed = dims == {expected} and missing == 0
    if not dims:
        message = f"{missing} rows missing embeddings"
    elif missing:
        message = f"dimensions {sorted(dims)}; {missing} rows missing embeddings"
    else:
        message = f"dimensions {sorted(dims)}; expected {expected}"
    return Check(name="uniform_dimensions", passed=passed, message=message)


def _check_finite(dataset: Dataset, column: str) -> Check:
    bad = 0
    for vector in _vectors(dataset, column):
        if vector is None:
            continue
        if any(not math.isfinite(value) for value in vector):
            bad += 1
    return Check(
        name="finite_values",
        passed=bad == 0,
        message="all embedding values are finite"
        if bad == 0
        else f"{bad} embeddings contain NaN or Inf",
    )


def _check_revision(job: Job, provenance_revision: str) -> Check:
    passed = bool(job.source.revision) and job.source.revision == provenance_revision
    return Check(
        name="source_revision",
        passed=passed,
        message=f"revision {job.source.revision}"
        if passed
        else "source revision missing or does not match provenance",
    )


def _check_model_metadata(job: Job, provenance: Provenance) -> Check:
    embedding = provenance.embedding
    passed = (
        embedding.provider == job.embedding.provider
        and embedding.model == job.embedding.model
        and embedding.dimensions == job.embedding.dimensions
    )
    return Check(
        name="model_metadata",
        passed=bool(passed),
        message="provider, model, and dimensions recorded"
        if passed
        else "embedding metadata missing or mismatched",
    )


def _check_license(job: Job, provenance_license: str | None) -> Check:
    passed = bool(job.source.license) and job.source.license == provenance_license
    return Check(
        name="license_inherited",
        passed=passed,
        message=f"license {job.source.license}"
        if passed
        else "source license missing; cannot inherit",
    )


def _diagnostics(dataset: Dataset, column: str) -> list[Diagnostic]:
    vectors = [vector for vector in _vectors(dataset, column) if vector is not None]
    if not vectors:
        return [
            Diagnostic(name="mean_vector_norm", message="no embeddings to diagnose"),
            Diagnostic(name="zero_vectors", message="0 zero vectors", value=0),
            Diagnostic(name="duplicate_embeddings", message="0 duplicates", value=0),
        ]
    norms = [math.sqrt(sum(value * value for value in vector)) for vector in vectors]
    mean_norm = sum(norms) / len(norms)
    zeros = sum(1 for norm in norms if norm == 0.0)
    counts = Counter(tuple(vector) for vector in vectors)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return [
        Diagnostic(
            name="mean_vector_norm",
            message=f"{mean_norm:.6f}",
            value=mean_norm,
        ),
        Diagnostic(
            name="zero_vectors",
            message=f"{zeros} zero vectors",
            value=zeros,
        ),
        Diagnostic(
            name="duplicate_embeddings",
            message=f"{duplicates} duplicate embeddings",
            value=duplicates,
        ),
    ]
