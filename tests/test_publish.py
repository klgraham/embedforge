from __future__ import annotations

from types import SimpleNamespace

from datasets import DatasetDict, load_from_disk
from huggingface_hub import DatasetCard

from embedforge.cache import EmbeddingCache
from embedforge.errors import EmbedForgeError
from embedforge.publish import (
    HuggingFaceHub,
    HuggingFacePublisher,
    build_dataset_card,
    merge_dataset_card,
    publish_job,
)
from embedforge.run import start_or_resume
from embedforge.shapes import Config, JobStatus, replace_job
from embedforge.store import JobStore
from tests.fakes import FakeDatasetSource, FakeEmbedder, FakeHub, hub_generated_readme


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


class _TimeoutThenPublicApi:
    def __init__(self) -> None:
        self.repo_exists_calls = 0

    def repo_exists(
        self, repo_id: str, *, repo_type: str | None = None, token: str | None = None
    ) -> bool:
        del repo_id, repo_type, token
        self.repo_exists_calls += 1
        if self.repo_exists_calls == 1:
            raise TimeoutError("timed out")
        return True

    def dataset_info(self, repo_id: str, token: str | None = None) -> SimpleNamespace:
        del repo_id, token
        return SimpleNamespace(private=False)


class _RecordingHuggingFaceHub(HuggingFaceHub):
    def __init__(self, api: _TimeoutThenPublicApi) -> None:
        super().__init__(api=api)
        self.uploads: list[str] = []

    def push_dataset(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.uploads.append("dataset")

    def read_text(self, *args: object, **kwargs: object) -> str:
        del args, kwargs
        self.uploads.append("read")
        return ""

    def upload_text(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.uploads.append("text")


def test_destination_lookup_timeout_does_not_upload(tmp_path) -> None:
    result, store, _source = _job(tmp_path)
    api = _TimeoutThenPublicApi()
    hub = _RecordingHuggingFaceHub(api)
    publisher = HuggingFacePublisher(hub=hub)
    try:
        publisher.publish(
            repo_id="owner/public",
            private=True,
            allow_public=False,
            revision=None,
            dataset_dir=store.output_dir(result.job.id),
            card="# card",
            provenance_yaml="embedforge: {version: 0.1.0}",
            config_name="default",
            token="unused",
        )
    except EmbedForgeError as exc:
        assert "inspect destination" in str(exc)
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected destination lookup timeout to abort publish")
    assert hub.uploads == []
    assert api.repo_exists_calls == 1


def test_inspect_destination_treats_false_as_not_found() -> None:
    class _MissingApi:
        def repo_exists(
            self, repo_id: str, *, repo_type: str | None = None, token: str | None = None
        ) -> bool:
            del repo_id, repo_type, token
            return False

        def dataset_info(self, repo_id: str, token: str | None = None) -> SimpleNamespace:
            del repo_id, token
            raise AssertionError("dataset_info should not run when repo_exists is false")

    info = HuggingFaceHub(api=_MissingApi()).inspect_destination("owner/new", "token")
    assert info.exists is False
    assert info.private is None


def test_publish_merges_hub_card_metadata_for_non_default_config(tmp_path) -> None:
    source = FakeDatasetSource(rows=[{"text": "alpha"}], config="fiqa")
    store = JobStore(tmp_path / "jobs")
    result = start_or_resume(
        "acme/fiqa",
        column="text",
        dataset_config="fiqa",
        config=Config(batch_size=8, concurrency=1, hf_namespace="klogram"),
        source=source,
        store=store,
        cache=EmbeddingCache(tmp_path / "embeddings"),
        embedder=FakeEmbedder(),
    )
    assert result.job.source.config == "fiqa"
    hub = FakeHub()
    published = publish_job(
        result.job.id,
        repo="owner/fiqa-emb",
        revision="v1",
        store=store,
        hub=hub,
        source=source,
    )
    readme = next(item for item in hub.uploads if item["kind"] == "README.md")
    content = readme["content"]
    assert isinstance(content, str)
    card = DatasetCard(content)
    configs = card.data["configs"]
    assert isinstance(configs, list)
    names = [item["config_name"] for item in configs]
    assert "fiqa" in names
    assert "corpus" in names
    infos = card.data["dataset_info"]
    assert isinstance(infos, list)
    info_names = [item["config_name"] for item in infos]
    assert "fiqa" in info_names
    assert "corpus" in info_names
    assert "EmbedForge" in content
    assert published.card.count("license:") >= 1


def test_merge_dataset_card_keeps_existing_config_mappings(tmp_path) -> None:
    result, store, _source = _job(tmp_path)
    job = store.load(result.job.id)
    ours = build_dataset_card(job, store.load_provenance(job.id).to_yaml())
    generated = hub_generated_readme("fiqa", ["train", "test"])
    merged = merge_dataset_card(generated, ours)
    card = DatasetCard(merged)
    names = [item["config_name"] for item in card.data["configs"]]
    assert names == ["fiqa", "corpus"]
    info_names = [item["config_name"] for item in card.data["dataset_info"]]
    assert "fiqa" in info_names
    assert "corpus" in info_names
    assert "data_files" in merged
    assert "EmbedForge" in merged
    assert job.source.license is not None
    assert card.data["license"] == job.source.license
