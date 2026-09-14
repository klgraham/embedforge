"""Build an embedding plan without making embedding API calls."""

from __future__ import annotations

import re

from embedforge.catalog import estimate_cost_usd, estimate_tokens, resolve_model
from embedforge.config import load_config
from embedforge.errors import EmbedForgeError
from embedforge.hfdata import DatasetRequest, DatasetSource, HuggingFaceDatasetSource
from embedforge.inspect import inspect_dataset
from embedforge.shapes import (
    Config,
    EmbeddingSettings,
    Estimates,
    InspectReport,
    OutputSpec,
    Plan,
    SourceRef,
    utc_now,
)


def default_output_repo(
    repository: str,
    provider: str,
    model: str,
    *,
    namespace: str | None,
) -> str:
    dataset = sanitize_repo_piece(repository.split("/")[-1])
    model_slug = sanitize_repo_piece(model.split("/")[-1])
    name = f"{dataset}-{provider}-{model_slug}"
    if namespace:
        return f"{namespace}/{name}"
    return name


def sanitize_repo_piece(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower())
    cleaned = cleaned.strip("-._")
    return cleaned or "dataset"


def build_plan(
    repository: str,
    *,
    column: str,
    config: Config | None = None,
    provider: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    dataset_config: str | None = None,
    split: str | None = None,
    output_column: str | None = None,
    limit: int | None = None,
    source: DatasetSource | None = None,
) -> Plan:
    """Describe the job. Must not call an embedding provider."""
    resolved = config or load_config()
    gateway = source or HuggingFaceDatasetSource()
    report = inspect_dataset(repository, config=dataset_config, split=split, source=gateway)
    if column not in {item.name for item in report.columns} and report.columns:
        raise EmbedForgeError(
            f"column {column!r} is not in the dataset; "
            f"candidates: {', '.join(report.candidate_text_columns) or 'none'}"
        )
    chosen_provider = provider or resolved.provider
    chosen_model = model or resolved.model
    chosen_dimensions = dimensions if dimensions is not None else resolved.dimensions
    info = resolve_model(chosen_provider, chosen_model, chosen_dimensions)
    stats = gateway.character_stats(
        DatasetRequest(
            repository=repository,
            config=report.config,
            split=report.split,
            revision=report.revision,
        ),
        column,
    )
    rows = stats.rows or report.estimated_rows
    if limit is not None:
        rows = min(rows, limit)
        if stats.rows:
            characters = int(stats.estimated_characters * (rows / stats.rows))
        else:
            characters = stats.estimated_characters
    else:
        characters = stats.estimated_characters
    tokens = estimate_tokens(characters)
    settings = EmbeddingSettings(
        provider=info.provider.value,
        model=info.model,
        dimensions=info.dimensions,
        column=output_column or resolved.output_column,
        source_columns=(column,),
        batch_size=resolved.batch_size,
        concurrency=resolved.concurrency,
    )
    output = OutputSpec(
        repo=default_output_repo(
            repository,
            settings.provider,
            settings.model,
            namespace=resolved.hf_namespace,
        ),
        column=settings.column,
    )
    return Plan(
        source=source_from_report(report),
        embedding=settings,
        output=output,
        estimates=Estimates(
            rows=rows,
            characters=characters,
            tokens=tokens,
            cost_usd=estimate_cost_usd(tokens, info.usd_per_million_tokens),
        ),
        created_at=utc_now(),
    )


def source_from_report(report: InspectReport) -> SourceRef:
    return SourceRef(
        repository=report.repository,
        revision=report.revision,
        config=report.config,
        split=report.split,
        license=report.license,
    )
