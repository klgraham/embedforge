from __future__ import annotations

import json
import sqlite3

import pytest

import embedforge.store as store_mod
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
    assert cache.get("openai", "text-embedding-3-small", 3, "hello") == pytest.approx(vector)
    assert key == cache_key("openai", "text-embedding-3-small", 3, "hello")
    assert cache.get("openai", "text-embedding-3-small", 3, "goodbye") is None
    info = cache.info()
    assert info["entries"] == 1
    assert info["legacy_entries"] == 0
    with sqlite3.connect(cache.database_path) as connection:
        stored_bytes = connection.execute(
            "SELECT length(embedding) FROM embeddings WHERE key = ?", (key,)
        ).fetchone()
    assert stored_bytes == (12,)
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
    store.append_embedding(job_id, 1, "k", status=RowStatus.SUCCESS, split="train")
    records = store.load_records(job_id)
    assert set(records) == {("train", 0), ("train", 1)}
    assert records[("train", 1)].embedding is None
    store.append_embedding(job_id, 2, "k", status=RowStatus.SUCCESS, split="train")
    records = store.load_records(job_id)
    assert set(records) == {("train", 0), ("train", 1), ("train", 2)}
    text = path.read_text()
    assert '{"index":{' not in text
    assert '"embedding"' not in text.splitlines()[-1]


def test_legacy_json_cache_migrates_to_sqlite(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings")
    key = cache_key("openai", "text-embedding-3-small", 3, "hello")
    path = cache.path_for(key)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "key": key,
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 3,
                "embedding": [0.1, 0.2, 0.3],
                "created_at": "2026-09-14T00:00:00Z",
            }
        )
    )

    first = cache.migrate_legacy()
    assert path.exists()
    result = cache.migrate_legacy(delete=True)

    assert first.scanned == 1
    assert first.imported == 1
    assert first.deleted == 0
    assert result.scanned == 1
    assert result.imported == 0
    assert result.existing == 1
    assert result.deleted == 1
    assert not path.exists()
    assert cache.get("openai", "text-embedding-3-small", 3, "hello") == pytest.approx(
        [0.1, 0.2, 0.3]
    )


def test_legacy_job_journal_compacts_to_cache_references(tmp_path) -> None:
    source = FakeDatasetSource(rows=[{"text": "hello"}])
    store = JobStore(tmp_path / "jobs")
    cache = EmbeddingCache(tmp_path / "embeddings")
    result = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=8, concurrency=1),
        source=source,
        store=store,
        cache=cache,
        embedder=FakeEmbedder(),
    )
    key = cache_key("openai", "text-embedding-3-small", 1536, "hello")
    vector = cache.get_by_key(key, dimensions=1536)
    assert vector is not None
    path = store.embeddings_path(result.job.id)
    path.write_text(
        json.dumps(
            {
                "split": "train",
                "index": 0,
                "status": "success",
                "cache_key": key,
                "embedding": vector,
            }
        )
        + "\n"
    )
    cache.clean()

    compacted = store.compact_journal(result.job.id, cache)

    assert compacted.bytes_after < compacted.bytes_before
    assert '"embedding"' not in path.read_text()
    assert cache.get_by_key(key, dimensions=1536) == pytest.approx(vector)
    repeated = store.compact_journal(result.job.id, cache)
    assert repeated.bytes_after == compacted.bytes_after


def test_interior_journal_corruption_is_still_reported(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs")
    job_id = "01TESTJOURNALCORRUPT0000000"
    (store.root / job_id).mkdir(parents=True)
    path = store.embeddings_path(job_id)
    path.write_text(
        "not-json\n"
        '{"split":"train","index":0,"status":"success","cache_key":"k","embedding":[1.0]}\n'
    )
    try:
        store.load_records(job_id)
    except EmbedForgeError as exc:
        assert "corrupt embeddings journal" in str(exc)
    else:
        raise AssertionError("expected interior journal corruption to be reported")
    try:
        store.append_embedding(job_id, 1, "k", status=RowStatus.SUCCESS, split="train")
    except EmbedForgeError as exc:
        assert "corrupt embeddings journal" in str(exc)
    else:
        raise AssertionError("expected append to refuse an interior-corrupt journal")


def test_repeated_appends_parse_journal_once(tmp_path, monkeypatch) -> None:
    parses = {"n": 0}
    original = store_mod._journal_loads

    def counted(line: str):
        parses["n"] += 1
        return original(line)

    monkeypatch.setattr(store_mod, "_journal_loads", counted)

    def append_n(count: int) -> int:
        parses["n"] = 0
        store = JobStore(tmp_path / f"jobs-{count}")
        job_id = f"scale-{count}"
        (store.root / job_id).mkdir(parents=True)
        store.embeddings_path(job_id).write_text(
            '{"split":"train","index":0,"status":"success","cache_key":"k","embedding":[1.0]}\n'
        )
        for index in range(1, count + 1):
            store.append_embedding(job_id, index, "k", status=RowStatus.SUCCESS, split="train")
        store.close_journal(job_id)
        return parses["n"]

    small_n = 80
    large_n = 160
    small = append_n(small_n)
    large = append_n(large_n)
    assert small <= 2
    assert large <= 2
    assert large / small <= 2


def test_run_does_not_reparse_journal_each_window(tmp_path, monkeypatch) -> None:
    parses = {"n": 0}
    original = store_mod._journal_loads

    def counted(line: str):
        parses["n"] += 1
        return original(line)

    monkeypatch.setattr(store_mod, "_journal_loads", counted)
    start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=8, concurrency=1),
        source=FakeDatasetSource(rows=[{"text": f"row-{index}"} for index in range(80)]),
        store=JobStore(tmp_path / "jobs"),
        cache=EmbeddingCache(tmp_path / "embeddings"),
        embedder=FakeEmbedder(),
    )
    assert parses["n"] == 0
