from __future__ import annotations

import embedforge.run as run_mod
from embedforge.cache import EmbeddingCache
from embedforge.run import start_or_resume
from embedforge.shapes import Config, JobStatus, RowStatus, replace_job
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
    store.save_job(replace_job(store.load(first.job.id), status=JobStatus.RUNNING))

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
    assert resumed.job.progress.failed == 0


def test_resume_retries_provider_failures(tmp_path) -> None:
    rows = [{"text": f"row-{index}"} for index in range(4)]
    source = FakeDatasetSource(rows=rows)
    store = JobStore(tmp_path / "jobs")
    cache = EmbeddingCache(tmp_path / "embeddings")
    embedder = FakeEmbedder(fail_on={"row-1", "row-2"})
    first = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=2, concurrency=1),
        source=source,
        store=store,
        cache=cache,
        embedder=embedder,
    )
    assert first.job.status is JobStatus.FAILED
    assert first.job.progress.failed == 2
    records = store.load_records(first.job.id)
    assert records[("train", 1)].status is RowStatus.FAILED
    assert records[("train", 2)].status is RowStatus.FAILED
    assert records[("train", 0)].status is RowStatus.SUCCESS

    embedder.fail_on.clear()
    calls_after_failure = len(embedder.calls)
    resumed = start_or_resume(
        None,
        resume_id=first.job.id,
        source=source,
        store=store,
        cache=cache,
        embedder=embedder,
    )
    assert resumed.job.status is JobStatus.COMPLETED
    retried = [text for batch in embedder.calls[calls_after_failure:] for text in batch]
    assert set(retried) == {"row-1", "row-2"}
    assert resumed.job.progress.failed == 0
    assert resumed.job.progress.embedded == 2
    assert store.load_records(first.job.id)[("train", 1)].status is RowStatus.SUCCESS
    assert store.load_records(first.job.id)[("train", 2)].status is RowStatus.SUCCESS


def test_duplicate_inputs_are_billed_once(tmp_path) -> None:
    source = FakeDatasetSource(rows=[{"text": "same"}, {"text": "same"}, {"text": "other"}])
    embedder = FakeEmbedder()
    result = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=8, concurrency=1),
        source=source,
        store=JobStore(tmp_path / "jobs"),
        cache=EmbeddingCache(tmp_path / "embeddings"),
        embedder=embedder,
    )
    assert result.job.status is JobStatus.COMPLETED
    assert embedder.calls == [["same", "other"]]
    assert result.job.progress.embedded == 3


def test_resume_repairs_torn_journal_and_finishes_remaining_work(tmp_path) -> None:
    rows = [{"text": f"row-{index}"} for index in range(3)]
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
    path = store.embeddings_path(first.job.id)
    first_line = path.read_text().splitlines()[0]
    path.write_text(first_line + "\n" + '{"index":')
    assert cache.clean() == 3
    store.save_job(replace_job(store.load(first.job.id), status=JobStatus.RUNNING))
    calls_before = len(embedder.calls)
    resumed = start_or_resume(
        None,
        resume_id=first.job.id,
        source=source,
        store=store,
        cache=cache,
        embedder=embedder,
    )
    assert resumed.job.status is JobStatus.COMPLETED
    records = store.load_records(first.job.id)
    assert set(records) == {("train", 0), ("train", 1), ("train", 2)}
    assert all(record.status is RowStatus.SUCCESS for record in records.values())
    retried = [text for batch in embedder.calls[calls_before:] for text in batch]
    assert set(retried) == {"row-1", "row-2"}
    store.append_embedding(first.job.id, 99, "x", [0.0], status=RowStatus.SUCCESS, split="train")
    assert ("train", 99) in store.load_records(first.job.id)


def test_result_fanout_scales_linearly_with_unique_inputs(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}
    original = run_mod._key

    def counted(job, text):
        calls["n"] += 1
        return original(job, text)

    monkeypatch.setattr(run_mod, "_key", counted)

    def embed_n(count: int) -> int:
        calls["n"] = 0
        rows = [{"text": f"unique-{index}"} for index in range(count)]
        start_or_resume(
            "acme/fiqa",
            column="text",
            config=Config(batch_size=16, concurrency=1),
            source=FakeDatasetSource(rows=rows),
            store=JobStore(tmp_path / f"jobs-{count}"),
            cache=EmbeddingCache(tmp_path / f"embeddings-{count}"),
            embedder=FakeEmbedder(),
        )
        return calls["n"]

    small_n = 80
    large_n = 160
    small = embed_n(small_n)
    large = embed_n(large_n)
    assert small <= small_n * 2
    assert large <= large_n * 2
    assert large / small < 2.5
