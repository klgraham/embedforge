from __future__ import annotations

import embedforge.plan as plan_mod
import embedforge.providers as providers_mod
from embedforge.errors import EmbedForgeError
from embedforge.plan import build_plan, default_output_repo
from embedforge.shapes import Config, Plan
from tests.fakes import FakeDatasetSource, FakeEmbedder


def test_plan_does_not_call_embedding_provider(monkeypatch) -> None:
    source = FakeDatasetSource(
        rows=[{"text": "query about apples"}, {"text": "query about oranges"}]
    )
    embedder = FakeEmbedder()

    def explode(provider: str) -> FakeEmbedder:
        raise AssertionError(f"plan must not construct embedder for {provider}")

    monkeypatch.setattr(providers_mod, "get_embedder", explode)
    monkeypatch.setattr(plan_mod, "load_config", lambda: Config(hf_namespace="klogram"))
    plan = build_plan(
        "acme/fiqa",
        column="text",
        config=Config(provider="openai", model="text-embedding-3-small", hf_namespace="klogram"),
        source=source,
    )
    assert embedder.calls == []
    assert plan.source.revision == "abc123def456"
    assert plan.embedding.provider == "openai"
    assert plan.embedding.model == "text-embedding-3-small"
    assert plan.embedding.dimensions == 1536
    assert plan.embedding.storage_dtype == "float32"
    assert plan.output.repo == "klogram/fiqa-openai-text-embedding-3-small-1536"
    assert plan.output.column == "embedding"
    assert plan.estimates.rows == 2
    assert plan.estimates.tokens > 0
    assert plan.estimates.cost_usd > 0


def test_plan_limit_reduces_estimates() -> None:
    source = FakeDatasetSource(rows=[{"text": "x" * 40} for _ in range(10)])
    full = build_plan("acme/fiqa", column="text", source=source)
    limited = build_plan("acme/fiqa", column="text", limit=2, source=source)
    assert limited.estimates.rows == 2
    assert limited.estimates.characters < full.estimates.characters
    assert limited.estimates.cost_usd < full.estimates.cost_usd


def test_plan_rejects_output_column_collision() -> None:
    source = FakeDatasetSource(rows=[{"text": "hello"}])
    try:
        build_plan("acme/fiqa", column="text", output_column="text", source=source)
    except EmbedForgeError as exc:
        assert "collides" in str(exc)
    else:
        raise AssertionError("expected colliding output column to fail before embedding")


def test_default_output_repo_normalizes_provider_independent_model() -> None:
    assert (
        default_output_repo(
            "BeIR/fiqa",
            "openai",
            "openai/text-embedding-3-small",
            dimensions=1536,
            namespace=None,
        )
        == "fiqa-openai-text-embedding-3-small-1536"
    )


def test_plan_output_repo_uses_requested_dimensions() -> None:
    source = FakeDatasetSource(rows=[{"text": "hello"}])
    plan = build_plan(
        "acme/fiqa",
        column="text",
        dimensions=512,
        config=Config(provider="openai", model="text-embedding-3-small"),
        source=source,
    )
    assert plan.embedding.dimensions == 512
    assert plan.output.repo == "fiqa-openai-text-embedding-3-small-512"


def test_plan_records_float16_storage_without_changing_embedding_dimensions() -> None:
    source = FakeDatasetSource(rows=[{"text": "hello"}])
    plan = build_plan(
        "acme/fiqa",
        column="text",
        config=Config(storage_dtype="float16"),
        source=source,
    )
    assert plan.embedding.storage_dtype == "float16"
    assert plan.embedding.dimensions == 1536


def test_legacy_plan_without_storage_dtype_defaults_to_float32() -> None:
    source = FakeDatasetSource(rows=[{"text": "hello"}])
    raw = build_plan("acme/fiqa", column="text", source=source).to_dict()
    raw["embedding"].pop("storage_dtype")

    restored = Plan.from_dict(raw)

    assert restored.embedding.storage_dtype == "float32"


def test_plan_rejects_unknown_storage_dtype() -> None:
    source = FakeDatasetSource(rows=[{"text": "hello"}])
    try:
        build_plan(
            "acme/fiqa",
            column="text",
            config=Config(storage_dtype="float64"),
            source=source,
        )
    except EmbedForgeError as exc:
        assert "unknown storage dtype" in str(exc)
    else:
        raise AssertionError("expected unknown storage dtype to fail")
