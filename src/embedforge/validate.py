"""Validate a staged embedding job against source and provenance rules."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

from embedforge.errors import EmbedForgeError
from embedforge.hfdata import (
    DatasetRequest,
    DatasetSource,
    HuggingFaceDatasetSource,
    LoadedDataset,
    embedding_input,
)
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
    staged = _load_staged(output)

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
        _check_row_counts(staged, loaded),
        _check_split(job, staged, loaded),
        _check_source_columns(staged, loaded.columns),
        _check_embedding_column(staged, job.embedding.column),
        _check_dimensions(
            staged,
            job.embedding.column,
            job.embedding.dimensions,
            job.embedding.source_columns[0],
        ),
        _check_finite(staged, job.embedding.column),
        _check_revision(job, provenance.source.revision),
        _check_model_metadata(job, provenance),
        _check_license(job, provenance.source.license),
        _check_config(job),
    ]
    diagnostics = _diagnostics(staged, job.embedding.column)
    result = ValidationResult(
        ok=all(check.passed for check in checks),
        checks=tuple(checks),
        diagnostics=tuple(diagnostics),
    )
    if mark_status and result.ok and job.status in {JobStatus.COMPLETED, JobStatus.VALIDATED}:
        job_store.save_job(replace_job(job, status=JobStatus.VALIDATED))
    return result


def _load_staged(output: object) -> DatasetDict:
    raw = load_from_disk(str(output))
    if isinstance(raw, DatasetDict):
        return raw
    if isinstance(raw, Dataset):
        name = getattr(raw, "split", None)
        split_name = name if isinstance(name, str) and name else "train"
        return DatasetDict({split_name: raw})
    raise EmbedForgeError("staged output must be a dataset or dataset dict")


def _iter_splits(staged: DatasetDict) -> list[tuple[str, Dataset]]:
    return [(str(name), staged[name]) for name in staged]


def _column(dataset: Dataset, name: str) -> list[Any]:
    if name not in dataset.column_names:
        return []
    return list(dataset[name])


def _check_row_counts(staged: DatasetDict, loaded: LoadedDataset) -> Check:
    expected = {item.name: len(item.rows) for item in loaded.splits}
    actual = {name: len(dataset) for name, dataset in _iter_splits(staged)}
    passed = actual == expected
    return Check(
        name="source_row_counts",
        passed=passed,
        message=f"output splits {actual}; source splits {expected}",
    )


def _check_split(job: Job, staged: DatasetDict, loaded: LoadedDataset) -> Check:
    actual = set(staged.keys())
    expected = set(loaded.split_names())
    if job.source.split is not None:
        expected = {job.source.split}
    passed = actual == expected and bool(actual)
    return Check(
        name="split_structure",
        passed=passed,
        message=f"staged splits {sorted(actual)}; expected {sorted(expected)}",
    )


def _check_source_columns(staged: DatasetDict, source_columns: tuple[str, ...]) -> Check:
    missing: list[str] = []
    for name, dataset in _iter_splits(staged):
        missing.extend(
            f"{name}.{column}" for column in source_columns if column not in dataset.column_names
        )
    passed = not missing
    return Check(
        name="source_columns",
        passed=passed,
        message="all source columns present"
        if passed
        else f"missing source columns: {', '.join(missing)}",
    )


def _check_embedding_column(staged: DatasetDict, column: str) -> Check:
    missing = [name for name, dataset in _iter_splits(staged) if column not in dataset.column_names]
    passed = not missing
    return Check(
        name="embedding_column",
        passed=passed,
        message=f"column {column!r} present"
        if passed
        else f"column {column!r} missing from {', '.join(missing)}",
    )


def _vectors(staged: DatasetDict, column: str) -> list[list[float] | None]:
    values: list[list[float] | None] = []
    for _name, dataset in _iter_splits(staged):
        if column not in dataset.column_names:
            continue
        for item in _column(dataset, column):
            if item is None:
                values.append(None)
            elif isinstance(item, list) and all(isinstance(num, (int, float)) for num in item):
                values.append([float(num) for num in item])
            else:
                values.append(None)
    return values


def _check_dimensions(
    staged: DatasetDict,
    column: str,
    expected: int,
    source_column: str,
) -> Check:
    missing_cols = [
        name for name, dataset in _iter_splits(staged) if column not in dataset.column_names
    ]
    if missing_cols:
        return Check(
            name="uniform_dimensions",
            passed=False,
            message="embedding column missing",
        )
    missing_source_cols = [
        name for name, dataset in _iter_splits(staged) if source_column not in dataset.column_names
    ]
    if missing_source_cols:
        return Check(
            name="uniform_dimensions",
            passed=False,
            message=f"source column {source_column!r} missing",
        )
    dims: set[int] = set()
    empty_nulls = 0
    missing = 0
    malformed = 0
    for _name, dataset in _iter_splits(staged):
        source_values = _column(dataset, source_column)
        embeddings = _column(dataset, column)
        for source_value, vector in zip(source_values, embeddings, strict=True):
            if vector is None:
                if embedding_input(source_value) == "":
                    empty_nulls += 1
                else:
                    missing += 1
            elif isinstance(vector, list) and all(isinstance(num, (int, float)) for num in vector):
                dims.add(len(vector))
            else:
                malformed += 1

    wrong_dimensions = dims - {expected}
    passed = not wrong_dimensions and missing == 0 and malformed == 0
    parts = (
        [f"dimensions {sorted(dims)}; expected {expected}"] if dims else ["no non-null embeddings"]
    )
    if empty_nulls:
        parts.append(f"{empty_nulls} null embeddings allowed for empty {source_column!r} values")
    if missing:
        parts.append(f"{missing} non-empty rows missing embeddings")
    if malformed:
        parts.append(f"{malformed} malformed embeddings")
    message = "; ".join(parts)
    return Check(name="uniform_dimensions", passed=passed, message=message)


def _check_finite(staged: DatasetDict, column: str) -> Check:
    bad = 0
    for vector in _vectors(staged, column):
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
        and embedding.storage_dtype == job.embedding.storage_dtype
    )
    return Check(
        name="model_metadata",
        passed=bool(passed),
        message="provider, model, dimensions, and storage dtype recorded"
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


def _check_config(job: Job) -> Check:
    passed = bool(job.source.config)
    return Check(
        name="source_config",
        passed=passed,
        message=f"config {job.source.config}" if passed else "source config was not recorded",
    )


def _diagnostics(staged: DatasetDict, column: str) -> list[Diagnostic]:
    vectors = [vector for vector in _vectors(staged, column) if vector is not None]
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
