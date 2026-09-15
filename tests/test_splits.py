from __future__ import annotations

from datasets import DatasetDict, load_from_disk

from embedforge.cache import EmbeddingCache
from embedforge.plan import build_plan
from embedforge.publish import publish_job
from embedforge.run import start_or_resume
from embedforge.shapes import Config
from embedforge.store import JobStore
from embedforge.validate import validate_job
from tests.fakes import FakeDatasetSource, FakeEmbedder, FakeHub


def test_selected_split_is_preserved_on_disk(tmp_path) -> None:
    source = FakeDatasetSource(
        split_data={
            "train": [{"text": "train-a"}],
            "test": [{"text": "test-a"}, {"text": "test-b"}],
        }
    )
    store = JobStore(tmp_path / "jobs")
    result = start_or_resume(
        "acme/fiqa",
        column="text",
        split="test",
        config=Config(batch_size=8, concurrency=1, hf_namespace="klogram"),
        source=source,
        store=store,
        cache=EmbeddingCache(tmp_path / "embeddings"),
        embedder=FakeEmbedder(),
    )
    staged = load_from_disk(str(result.output_path))
    assert isinstance(staged, DatasetDict)
    assert list(staged.keys()) == ["test"]
    assert len(staged["test"]) == 2
    embedding_feature = staged["test"].features["embedding"]
    assert embedding_feature.length == result.job.embedding.dimensions
    assert embedding_feature.feature.dtype == "float32"
    assert result.job.source.split == "test"
    assert result.job.source.config == "default"
    report = validate_job(result.job.id, store=store, source=source)
    assert report.ok
    split_check = next(check for check in report.checks if check.name == "split_structure")
    assert split_check.passed


def test_all_splits_are_transformed_when_split_omitted(tmp_path) -> None:
    source = FakeDatasetSource(
        split_data={
            "train": [{"text": "train-a"}],
            "test": [{"text": "test-a"}],
        }
    )
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
    staged = load_from_disk(str(result.output_path))
    assert isinstance(staged, DatasetDict)
    assert set(staged.keys()) == {"train", "test"}
    assert result.job.source.split is None
    plan = build_plan("acme/fiqa", column="text", source=source)
    assert plan.source.split is None
    assert plan.estimates.rows == 2
    hub = FakeHub()
    published = publish_job(result.job.id, store=store, hub=hub, source=source)
    dataset = next(item for item in hub.uploads if item["kind"] == "dataset")
    raw_splits = dataset["splits"]
    assert isinstance(raw_splits, list)
    assert set(raw_splits) == {"train", "test"}
    assert dataset["config_name"] == "default"
    assert published.private is True


def test_float16_storage_is_explicit_and_fixed_width(tmp_path) -> None:
    source = FakeDatasetSource(rows=[{"text": "one"}, {"text": "two"}])
    store = JobStore(tmp_path / "jobs")

    result = start_or_resume(
        "acme/fiqa",
        column="text",
        config=Config(batch_size=8, concurrency=1, storage_dtype="float16"),
        source=source,
        store=store,
        cache=EmbeddingCache(tmp_path / "embeddings"),
        embedder=FakeEmbedder(),
    )

    staged = load_from_disk(str(result.output_path))
    assert isinstance(staged, DatasetDict)
    embedding_feature = staged["train"].features["embedding"]
    assert embedding_feature.length == result.job.embedding.dimensions
    assert embedding_feature.feature.dtype == "float16"
    assert result.job.embedding.storage_dtype == "float16"
    assert store.load_provenance(result.job.id).embedding.storage_dtype == "float16"
