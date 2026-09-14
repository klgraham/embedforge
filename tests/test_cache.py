from __future__ import annotations

from embedforge.cache import EmbeddingCache, cache_key
from embedforge.errors import EmbedForgeError
from embedforge.run import start_or_resume
from embedforge.shapes import Config, RowStatus
from embedforge.store import JobStore
from tests.fakes import FakeDatasetSource, FakeEmbedder


def test_cache_key_identity() -> None:
    base = cache_key("openai", "text-embedding-3-small", 1536, "hello")
    assert cache_key("openai", "text-embedding-3-small", 1536, "hello") == base
    assert cache_key("openrouter", "text-embedding-3-small", 1536, "hello") != base
    assert cache_key("openai", "text-embedding-3-large", 1536, "hello") != base
    assert cache_key("openai", "text-embedding-3-small", 512, "hello") != base
    assert cache_key("openai", "text-embedding-3-small", 1536, "HELLO") != base


def test_cache_hit_avoids_second_api_call(tmp_path) -> None:
    rows = [{"text": "same text"}, {"text": "other text"}]
    source = FakeDatasetSource(rows=rows)
    store = JobStore(tmp_path / "jobs")
    cache = EmbeddingCache(tmp_path / "embeddings")
    embedder = FakeEmbedder()
    first = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=8, concurrency=1),
        source=source,
        store=store,
        cache=cache,
        embedder=embedder,
    )
    assert first.job.progress.embedded == 2
    assert len(cache.list_entries()) == 2

    second = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=8, concurrency=1),
        source=source,
        store=store,
        cache=cache,
        embedder=embedder,
    )
    assert second.job.id != first.job.id
    assert second.job.progress.embedded == 0
    assert second.job.progress.skipped == 2
    assert len(embedder.calls) == 1


def test_cache_put_get_roundtrip(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings")
    vector = [0.1, 0.2, 0.3]
    key = cache.put("openai", "text-embedding-3-small", 3, "hello", vector)
    assert cache.get("openai", "text-embedding-3-small", 3, "hello") == vector
    assert key == cache_key("openai", "text-embedding-3-small", 3, "hello")
    assert cache.get("openai", "text-embedding-3-small", 3, "goodbye") is None
    info = cache.info()
    assert info["entries"] == 1
    assert cache.clean() == 1
    assert cache.list_entries() == []


def test_journal_recovers_incomplete_trailing_record(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs")
    job_id = "01TESTJOURNAL000000000000"
    (store.root / job_id).mkdir(parents=True)
    path = store.embeddings_path(job_id)
    path.write_text(
        '{"split":"train","index":0,"status":"success","cache_key":"k","embedding":[1.0]}\n'
        '{"split":"train","index":1,"status":"success","cache_key":"k","embedding":[2.0'
    )
    records = store.load_records(job_id)
    assert set(records) == {("train", 0)}
    assert records[("train", 0)].status is RowStatus.SUCCESS
    assert records[("train", 0)].embedding == [1.0]


def test_append_truncates_torn_journal_before_write(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs")
    job_id = "01TESTJOURNALAPPEND00000000"
    (store.root / job_id).mkdir(parents=True)
    path = store.embeddings_path(job_id)
    path.write_text(
        '{"split":"train","index":0,"status":"success","cache_key":"k","embedding":[1.0]}\n'
        '{"index":'
    )
    store.load_records(job_id)
    store.append_embedding(job_id, 1, "k", [2.0], status=RowStatus.SUCCESS, split="train")
    records = store.load_records(job_id)
    assert set(records) == {("train", 0), ("train", 1)}
    assert records[("train", 1)].embedding == [2.0]
    store.append_embedding(job_id, 2, "k", [3.0], status=RowStatus.SUCCESS, split="train")
    records = store.load_records(job_id)
    assert set(records) == {("train", 0), ("train", 1), ("train", 2)}
    text = path.read_text()
    assert '{"index":{' not in text


def test_interior_journal_corruption_is_still_reported(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs")
    job_id = "01TESTJOURNALCORRUPT0000000"
    (store.root / job_id).mkdir(parents=True)
    path = store.embeddings_path(job_id)
    path.write_text(
        'not-json\n'
        '{"split":"train","index":0,"status":"success","cache_key":"k","embedding":[1.0]}\n'
    )
    try:
        store.load_records(job_id)
    except EmbedForgeError as exc:
        assert "corrupt embeddings journal" in str(exc)
    else:
        raise AssertionError("expected interior journal corruption to be reported")
    try:
        store.append_embedding(job_id, 1, "k", [2.0], status=RowStatus.SUCCESS, split="train")
    except EmbedForgeError as exc:
        assert "corrupt embeddings journal" in str(exc)
    else:
        raise AssertionError("expected append to refuse an interior-corrupt journal")
