from __future__ import annotations

from datasets import DatasetDict, load_from_disk

from embedforge.cache import EmbeddingCache
from embedforge.errors import EmbedForgeError
from embedforge.publish import HuggingFacePublisher, build_dataset_card, publish_job
from embedforge.run import start_or_resume
from embedforge.shapes import Config, JobStatus, replace_job
from embedforge.store import JobStore
from tests.fakes import FakeDatasetSource, FakeEmbedder, FakeHub


def _job(tmp_path, *, split: str | None = None, rows=None, split_data=None):
    source = FakeDatasetSource(
        rows=rows or [{"text": "alpha"}, {"text": "beta"}],
        split_data=split_data,
    )
    store = JobStore(tmp_path / "jobs")
    result = start_or_resume(
        "acme/fiqa",
        column="text",
        split=split,
        config=Config(batch_size=8, concurrency=1, hf_namespace="klogram"),
        source=source,
        store=store,
        cache=EmbeddingCache(tmp_path / "embeddings"),
        embedder=FakeEmbedder(),
    )
    return result, store, source


def test_publish_defaults_private_and_records_provenance(tmp_path) -> None:
    result, store, source = _job(tmp_path)
    hub = FakeHub()
    published = publish_job(result.job.id, store=store, hub=hub, source=source)
    assert published.private is True
    dataset_uploads = [item for item in hub.uploads if item["kind"] == "dataset"]
    assert len(dataset_uploads) == 1
    assert dataset_uploads[0]["private_flag"] is True
    assert dataset_uploads[0]["actual_private"] is True
    assert dataset_uploads[0]["config_name"] == "default"
    card = published.card
    assert "abc123def456" in card
    assert "mit" in card
    assert "https://huggingface.co/datasets/acme/fiqa" in card
    yaml_uploads = [item for item in hub.uploads if item["kind"] == "embedforge.yaml"]
    assert "abc123def456" in str(yaml_uploads[0]["content"])


def test_publish_rejects_existing_public_without_explicit_public(tmp_path) -> None:
    result, store, source = _job(tmp_path)
    hub = FakeHub(repos={"klogram/fiqa-openai-text-embedding-3-small": {"private": False}})
    try:
        publish_job(result.job.id, store=store, hub=hub, source=source)
    except EmbedForgeError as exc:
        assert "is public" in str(exc)
        assert "--public" in str(exc)
    else:
        raise AssertionError("expected public destination to be rejected")
    assert hub.uploads == []


def test_publish_allows_existing_public_with_explicit_public(tmp_path) -> None:
    result, store, source = _job(tmp_path)
    repo = "owner/public-dest"
    hub = FakeHub(repos={repo: {"private": False}})
    published = publish_job(
        result.job.id,
        repo=repo,
        private=False,
        allow_public=True,
        store=store,
        hub=hub,
        source=source,
    )
    assert published.private is False
    assert hub.repos[repo]["private"] is False


def test_publish_existing_private_stays_private(tmp_path) -> None:
    result, store, source = _job(tmp_path)
    repo = "owner/private-dest"
    hub = FakeHub(repos={repo: {"private": True}})
    published = publish_job(result.job.id, repo=repo, store=store, hub=hub, source=source)
    assert published.private is True
    dataset = next(item for item in hub.uploads if item["kind"] == "dataset")
    assert dataset["actual_private"] is True


def test_publish_passes_revision_to_dataset_and_metadata(tmp_path) -> None:
    result, store, source = _job(tmp_path)
    hub = FakeHub()
    published = publish_job(
        result.job.id,
        repo="owner/custom",
        allow_public=True,
        private=False,
        revision="v1",
        store=store,
        hub=hub,
        source=source,
    )
    assert published.revision == "v1"
    kinds = {item["kind"] for item in hub.uploads}
    assert kinds == {"dataset", "README.md", "embedforge.yaml"}
    assert {item["revision"] for item in hub.uploads} == {"v1"}


def test_publish_rejects_failed_job(tmp_path) -> None:
    result, store, source = _job(tmp_path)
    store.save_job(replace_job(store.load(result.job.id), status=JobStatus.FAILED, error="nope"))
    hub = FakeHub()
    try:
        publish_job(result.job.id, store=store, hub=hub, source=source)
    except EmbedForgeError as exc:
        assert "completed job" in str(exc)
    else:
        raise AssertionError("expected failed job to be rejected")
    assert hub.uploads == []


def test_publish_rejects_invalid_output(tmp_path) -> None:
    source = FakeDatasetSource(rows=[{"text": "only"}], license=None)
    store = JobStore(tmp_path / "jobs")
    result = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=4, concurrency=1, hf_namespace="klogram"),
        source=source,
        store=store,
        cache=EmbeddingCache(tmp_path / "embeddings"),
        embedder=FakeEmbedder(),
    )
    assert result.job.status is JobStatus.COMPLETED
    hub = FakeHub()
    try:
        publish_job(result.job.id, store=store, hub=hub, source=source)
    except EmbedForgeError as exc:
        assert "validation" in str(exc)
    else:
        raise AssertionError("expected invalid job to be rejected")
    assert hub.uploads == []
    assert store.load(result.job.id).status is JobStatus.COMPLETED


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


def test_publisher_reports_actual_visibility_not_requested_flag(tmp_path) -> None:
    result, store, _source = _job(tmp_path)
    hub = FakeHub(repos={"owner/open": {"private": False}})
    publisher = HuggingFacePublisher(hub=hub)
    _url, actual_private = publisher.publish(
        repo_id="owner/open",
        private=True,
        allow_public=True,
        revision="v1",
        dataset_dir=store.output_dir(result.job.id),
        card="# card",
        provenance_yaml="embedforge: {version: 0.1.0}",
        config_name="default",
        token="unused",
    )
    assert actual_private is False
    staged = load_from_disk(str(store.output_dir(result.job.id)))
    assert isinstance(staged, DatasetDict)
    assert list(staged.keys())
