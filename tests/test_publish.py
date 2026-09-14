from __future__ import annotations

from embedforge.cache import EmbeddingCache
from embedforge.publish import build_dataset_card, publish_job
from embedforge.run import start_or_resume
from embedforge.shapes import Config
from embedforge.store import JobStore
from tests.fakes import FakeDatasetSource, FakeEmbedder, FakePublisher


def _job(tmp_path):
    source = FakeDatasetSource(rows=[{"text": "alpha"}, {"text": "beta"}])
    store = JobStore(tmp_path / "jobs")
    result = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=8, concurrency=1, hf_namespace="klogram"),
        source=source,
        store=store,
        cache=EmbeddingCache(tmp_path / "embeddings"),
        embedder=FakeEmbedder(),
    )
    return result, store, source


def test_publish_defaults_private_and_records_provenance(tmp_path) -> None:
    result, store, _source = _job(tmp_path)
    publisher = FakePublisher()
    published = publish_job(result.job.id, store=store, publisher=publisher)
    assert published.private is True
    assert len(publisher.calls) == 1
    call = publisher.calls[0]
    assert call["private"] is True
    assert call["repo_id"] == "klogram/fiqa-openai-text-embedding-3-small"
    card = str(call["card"])
    assert "abc123def456" in card
    assert "mit" in card
    assert "https://huggingface.co/datasets/acme/fiqa" in card
    assert "text-embedding-3-small" in card
    yaml = str(call["provenance_yaml"])
    assert "abc123def456" in yaml
    assert "openai" in yaml


def test_publish_public_flag(tmp_path) -> None:
    result, store, _source = _job(tmp_path)
    publisher = FakePublisher()
    published = publish_job(
        result.job.id,
        repo="owner/custom",
        private=False,
        revision="v1",
        store=store,
        publisher=publisher,
    )
    assert published.private is False
    assert published.revision == "v1"
    assert publisher.calls[0]["private"] is False
    assert publisher.calls[0]["repo_id"] == "owner/custom"
    assert publisher.calls[0]["revision"] == "v1"


def test_run_does_not_publish(tmp_path) -> None:
    result, store, _source = _job(tmp_path)
    assert result.job.published_repo is None
    assert store.load(result.job.id).published_repo is None


def test_dataset_card_names_source_revision(tmp_path) -> None:
    result, store, _source = _job(tmp_path)
    job = store.load(result.job.id)
    card = build_dataset_card(job, store.load_provenance(job.id).to_yaml())
    assert f"`{job.source.revision}`" in card
    assert job.source.license is not None
    assert job.source.license in card
