from __future__ import annotations

from embedforge.cache import EmbeddingCache
from embedforge.run import start_or_resume
from embedforge.shapes import Config, JobProgress, JobStatus, replace_job
from embedforge.store import JobStore
from tests.fakes import FakeDatasetSource, FakeEmbedder


def test_resume_skips_already_written_rows(tmp_path) -> None:
    rows = [{"text": f"row-{index}"} for index in range(5)]
    source = FakeDatasetSource(rows=rows)
    store = JobStore(tmp_path / "jobs")
    cache = EmbeddingCache(tmp_path / "embeddings")
    embedder = FakeEmbedder()
    first = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=2, concurrency=1),
        source=source,
        store=store,
        cache=cache,
        embedder=embedder,
    )
    assert first.job.status is JobStatus.COMPLETED
    first_calls = list(embedder.calls)

    job = store.load(first.job.id)
    store.save_job(
        replace_job(
            job,
            status=JobStatus.RUNNING,
            progress=JobProgress(next_index=0),
        )
    )
    resumed = start_or_resume(
        None,
        resume_id=first.job.id,
        source=source,
        store=store,
        cache=cache,
        embedder=embedder,
    )
    assert resumed.job.status is JobStatus.COMPLETED
    assert embedder.calls == first_calls
    assert resumed.job.progress.skipped == 5
    assert resumed.job.progress.embedded == 0


def test_partial_job_resumes_remaining_rows(tmp_path) -> None:
    rows = [{"text": f"row-{index}"} for index in range(4)]
    source = FakeDatasetSource(rows=rows)
    store = JobStore(tmp_path / "jobs")
    cache = EmbeddingCache(tmp_path / "embeddings")
    embedder = FakeEmbedder()
    first = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=2, concurrency=1),
        source=source,
        store=store,
        cache=cache,
        embedder=embedder,
    )
    embeddings = store.load_embeddings(first.job.id)
    cache.clean()
    store.embeddings_path(first.job.id).write_text("")
    for index in (0, 1):
        store.append_embedding(first.job.id, index, "kept", embeddings[index])
    store.save_job(
        replace_job(
            first.job,
            status=JobStatus.FAILED,
            progress=JobProgress(embedded=2, skipped=0, failed=0, next_index=2),
            error="interrupted",
        )
    )
    calls_after_first = len(embedder.calls)
    resumed = start_or_resume(
        None,
        resume_id=first.job.id,
        source=source,
        store=store,
        cache=cache,
        embedder=embedder,
    )
    assert resumed.job.status is JobStatus.COMPLETED
    new_texts = [text for batch in embedder.calls[calls_after_first:] for text in batch]
    assert new_texts == ["row-2", "row-3"]
    assert set(store.load_embeddings(first.job.id)) == {0, 1, 2, 3}
