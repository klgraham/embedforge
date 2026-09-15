from __future__ import annotations

from embedforge.cache import EmbeddingCache
from embedforge.run import start_or_resume
from embedforge.shapes import Config, JobStatus
from embedforge.store import JobStore
from embedforge.validate import validate_job
from tests.fakes import FakeDatasetSource, FakeEmbedder


def _run_job(tmp_path, rows, **kwargs):
    source = FakeDatasetSource(rows=rows)
    store = JobStore(tmp_path / "jobs")
    cache = EmbeddingCache(tmp_path / "embeddings")
    embedder = FakeEmbedder()
    result = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=8, concurrency=1, hf_namespace="klogram"),
        source=source,
        store=store,
        cache=cache,
        embedder=embedder,
        **kwargs,
    )
    return result, store, source


def test_validate_accepts_complete_job(tmp_path) -> None:
    result, store, source = _run_job(
        tmp_path, [{"text": "one"}, {"text": "two"}, {"text": "three"}]
    )
    report = validate_job(result.job.id, store=store, source=source)
    assert report.ok
    names = {check.name: check.passed for check in report.checks}
    assert names["source_row_counts"]
    assert names["embedding_column"]
    assert names["uniform_dimensions"]
    assert names["finite_values"]
    assert names["source_revision"]
    assert names["model_metadata"]
    assert names["license_inherited"]
    assert {item.name for item in report.diagnostics} >= {
        "mean_vector_norm",
        "zero_vectors",
        "duplicate_embeddings",
    }
    assert store.load(result.job.id).status is JobStatus.VALIDATED


def test_validate_accepts_null_embeddings_for_empty_source_values(tmp_path) -> None:
    result, store, source = _run_job(
        tmp_path,
        [{"text": ""}, {"text": None}, {"text": "embedded"}],
    )

    report = validate_job(result.job.id, store=store, source=source)

    assert report.ok
    dimensions = next(check for check in report.checks if check.name == "uniform_dimensions")
    assert dimensions.passed
    assert "2 null embeddings allowed for empty 'text' values" in dimensions.message
    assert result.job.progress.skipped == 2


def test_validate_accepts_dataset_with_only_empty_source_values(tmp_path) -> None:
    result, store, source = _run_job(tmp_path, [{"text": ""}, {"text": None}])

    report = validate_job(result.job.id, store=store, source=source)

    assert report.ok
    dimensions = next(check for check in report.checks if check.name == "uniform_dimensions")
    assert dimensions.message.startswith("no non-null embeddings")


def test_validate_rejects_null_embedding_for_non_empty_source_value(tmp_path) -> None:
    source = FakeDatasetSource(rows=[{"text": "provider failure"}])
    store = JobStore(tmp_path / "jobs")
    result = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=4, concurrency=1),
        source=source,
        store=store,
        cache=EmbeddingCache(tmp_path / "embeddings"),
        embedder=FakeEmbedder(fail_on={"provider failure"}),
    )

    report = validate_job(result.job.id, store=store, source=source)

    assert not report.ok
    dimensions = next(check for check in report.checks if check.name == "uniform_dimensions")
    assert not dimensions.passed
    assert "1 non-empty rows missing embeddings" in dimensions.message


def test_validate_fails_when_license_missing(tmp_path) -> None:
    source = FakeDatasetSource(rows=[{"text": "only"}], license=None)
    store = JobStore(tmp_path / "jobs")
    result = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=4, concurrency=1),
        source=source,
        store=store,
        cache=EmbeddingCache(tmp_path / "embeddings"),
        embedder=FakeEmbedder(),
    )
    report = validate_job(result.job.id, store=store, source=source)
    assert not report.ok
    license_check = next(check for check in report.checks if check.name == "license_inherited")
    assert not license_check.passed
