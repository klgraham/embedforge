"""Hugging Face dataset inspection and row loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from datasets import (
    get_dataset_config_names,
    load_dataset,
    load_dataset_builder,
)
from huggingface_hub import dataset_info

from embedforge.errors import EmbedForgeError
from embedforge.shapes import ColumnInfo, InspectReport, SourceRef

SAMPLE_SIZE = 256
TEXT_DTYPES = {"string", "large_string", "utf8"}


@dataclass(frozen=True)
class DatasetRequest:
    repository: str
    config: str | None = None
    split: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class CharStats:
    rows: int
    estimated_characters: int
    sampled_rows: int


@dataclass(frozen=True)
class LoadedSplit:
    name: str
    rows: list[dict[str, object]]
    columns: tuple[str, ...]


@dataclass(frozen=True)
class LoadedDataset:
    splits: tuple[LoadedSplit, ...]
    source: SourceRef

    @property
    def columns(self) -> tuple[str, ...]:
        return self.splits[0].columns if self.splits else ()

    def split_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.splits)

    def rows_for(self, name: str) -> list[dict[str, object]]:
        for item in self.splits:
            if item.name == name:
                return item.rows
        return []


class DatasetSource(Protocol):
    def inspect(self, request: DatasetRequest) -> InspectReport: ...

    def character_stats(self, request: DatasetRequest, column: str) -> CharStats: ...

    def load_rows(self, request: DatasetRequest, *, limit: int | None) -> LoadedDataset: ...


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# huggingface_hub.whoami is imported at use site in publish; token helper stays here.


def candidate_text_columns(columns: tuple[ColumnInfo, ...]) -> tuple[str, ...]:
    return tuple(column.name for column in columns if column.dtype.lower() in TEXT_DTYPES)


class HuggingFaceDatasetSource:
    def inspect(self, request: DatasetRequest) -> InspectReport:
        token = hf_token()
        hub = dataset_info(
            request.repository,
            revision=request.revision,
            token=token,
        )
        revision = request.revision or getattr(hub, "sha", None) or "unknown"
        license_name = _hub_license(hub)
        configs = _config_names(request.repository, token=token, revision=request.revision)
        config_name = request.config or _default_config(configs)
        builder = load_dataset_builder(
            request.repository,
            name=config_name,
            revision=request.revision,
            token=token,
        )
        info = builder.info
        if license_name is None:
            raw_license = getattr(info, "license", None)
            license_name = (raw_license or None) if isinstance(raw_license, str) else None
        splits = _split_counts(info)
        split_name = request.split or _default_split(splits)
        columns = _columns_from_features(getattr(info, "features", None))
        candidates = candidate_text_columns(columns)
        estimated_rows = splits.get(split_name, 0) if split_name else 0
        estimated_characters = 0
        if candidates and split_name:
            stats = self.character_stats(
                DatasetRequest(
                    repository=request.repository,
                    config=config_name,
                    split=split_name,
                    revision=revision if revision != "unknown" else request.revision,
                ),
                candidates[0],
            )
            estimated_rows = stats.rows or estimated_rows
            estimated_characters = stats.estimated_characters
        return InspectReport(
            repository=request.repository,
            revision=str(revision),
            license=license_name,
            config=config_name,
            split=split_name,
            configs=tuple(configs),
            splits=splits,
            columns=columns,
            candidate_text_columns=candidates,
            estimated_rows=estimated_rows,
            estimated_characters=estimated_characters,
        )

    def character_stats(self, request: DatasetRequest, column: str) -> CharStats:
        token = hf_token()
        split = request.split
        if split is None:
            raise EmbedForgeError("split is required to estimate characters")
        dataset = load_dataset(
            request.repository,
            name=request.config,
            split=split,
            revision=request.revision,
            token=token,
            streaming=True,
        )
        sampled_rows = 0
        sampled_chars = 0
        for row in dataset:
            sampled_rows += 1
            sampled_chars += _text_length(row.get(column) if isinstance(row, dict) else None)
            if sampled_rows >= SAMPLE_SIZE:
                break
        total_rows = _split_row_count(request, token)
        if sampled_rows == 0:
            return CharStats(rows=total_rows, estimated_characters=0, sampled_rows=0)
        estimated = int((sampled_chars / sampled_rows) * (total_rows or sampled_rows))
        return CharStats(
            rows=total_rows or sampled_rows,
            estimated_characters=estimated,
            sampled_rows=sampled_rows,
        )

    def load_rows(self, request: DatasetRequest, *, limit: int | None) -> LoadedDataset:
        token = hf_token()
        report = self.inspect(request)
        config_name = request.config or report.config
        revision = request.revision or report.revision
        if request.split is not None:
            names = (request.split,)
        elif report.splits:
            names = tuple(report.splits)
        elif report.split:
            names = (report.split,)
        else:
            raise EmbedForgeError("dataset has no splits to load")
        loaded: list[LoadedSplit] = []
        for split_name in names:
            dataset = load_dataset(
                request.repository,
                name=config_name,
                split=split_name,
                revision=revision,
                token=token,
            )
            if limit is not None:
                available = len(dataset)
                dataset = dataset.select(range(min(limit, available)))
            columns = tuple(str(name) for name in dataset.column_names)
            rows: list[dict[str, object]] = []
            for row in dataset:
                if isinstance(row, dict):
                    rows.append(dict(row))
            loaded.append(LoadedSplit(name=split_name, rows=rows, columns=columns))
        return LoadedDataset(
            splits=tuple(loaded),
            source=SourceRef(
                repository=request.repository,
                revision=report.revision,
                config=config_name,
                split=request.split,
                license=report.license,
            ),
        )


def _hub_license(hub: object) -> str | None:
    card = getattr(hub, "card_data", None) or getattr(hub, "cardData", None)
    if card is None:
        return None
    raw = getattr(card, "license", None)
    if raw is None and isinstance(card, dict):
        raw = card.get("license")
    if isinstance(raw, str):
        return raw or None
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        return raw[0] or None
    return None


def _config_names(repository: str, *, token: str | None, revision: str | None) -> list[str]:
    try:
        names = get_dataset_config_names(repository, token=token, revision=revision)
    except TypeError:
        names = get_dataset_config_names(repository, token=token)
    if not names:
        return ["default"]
    return [str(name) for name in names]


def _default_config(configs: list[str]) -> str | None:
    if not configs:
        return None
    if "default" in configs:
        return "default"
    return configs[0]


def _split_counts(info: object) -> dict[str, int]:
    splits = getattr(info, "splits", None)
    if not splits:
        return {}
    counts: dict[str, int] = {}
    items = splits.items() if hasattr(splits, "items") else []
    for name, spec in items:
        num = getattr(spec, "num_examples", None)
        counts[str(name)] = int(num) if isinstance(num, int) else 0
    return counts


def _default_split(splits: dict[str, int]) -> str | None:
    if not splits:
        return None
    if "train" in splits:
        return "train"
    return next(iter(splits))


def _columns_from_features(features: object) -> tuple[ColumnInfo, ...]:
    items = getattr(features, "items", None)
    if features is None or not callable(items):
        return ()
    columns: list[ColumnInfo] = []
    for name, feature in items():
        dtype = getattr(feature, "dtype", None)
        if not isinstance(dtype, str):
            dtype = type(feature).__name__
        columns.append(ColumnInfo(name=str(name), dtype=dtype))
    return tuple(columns)


def _split_row_count(request: DatasetRequest, token: str | None) -> int:
    builder = load_dataset_builder(
        request.repository,
        name=request.config,
        revision=request.revision,
        token=token,
    )
    splits = _split_counts(builder.info)
    if request.split and request.split in splits:
        return splits[request.split]
    return 0


def _text_length(value: object) -> int:
    return len(embedding_input(value))


def embedding_input(value: object) -> str:
    """Normalize one source value to the text sent to an embedding provider."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
