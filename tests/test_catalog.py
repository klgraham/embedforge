from __future__ import annotations

from embedforge.cache import EmbeddingCache
from embedforge.catalog import api_dimensions
from embedforge.run import start_or_resume
from embedforge.shapes import Config
from embedforge.store import JobStore
from tests.fakes import FakeDatasetSource, FakeEmbedder


def test_ada_does_not_send_dimensions_parameter() -> None:
    assert api_dimensions("text-embedding-ada-002", 1536) is None
    assert api_dimensions("openai/text-embedding-ada-002", 1536) is None
    assert api_dimensions("text-embedding-3-small", 1536) == 1536
    assert api_dimensions("text-embedding-3-small", 512) == 512


def test_run_omits_dimensions_for_ada(tmp_path) -> None:
    embedder = FakeEmbedder(dimensions=1536)
    result = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(model="text-embedding-ada-002", batch_size=8, concurrency=1),
        source=FakeDatasetSource(rows=[{"text": "hello"}]),
        store=JobStore(tmp_path / "jobs"),
        cache=EmbeddingCache(tmp_path / "embeddings"),
        embedder=embedder,
    )
    assert result.job.embedding.dimensions == 1536
    assert embedder.seen_dimensions == [None]
