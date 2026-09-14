"""Inspect a source Hugging Face dataset without embedding it."""

from __future__ import annotations

from embedforge.hfdata import DatasetRequest, DatasetSource, HuggingFaceDatasetSource
from embedforge.shapes import InspectReport


def inspect_dataset(
    repository: str,
    *,
    config: str | None = None,
    split: str | None = None,
    source: DatasetSource | None = None,
) -> InspectReport:
    gateway = source or HuggingFaceDatasetSource()
    return gateway.inspect(DatasetRequest(repository=repository, config=config, split=split))
