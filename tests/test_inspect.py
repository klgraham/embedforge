from __future__ import annotations

from embedforge.hfdata import DatasetRequest
from embedforge.inspect import inspect_dataset
from tests.fakes import FakeDatasetSource


def test_inspect_reports_revision_license_and_text_columns() -> None:
    source = FakeDatasetSource(
        rows=[
            {"_id": "1", "text": "alpha " * 20},
            {"_id": "2", "text": "beta " * 10},
        ]
    )
    report = inspect_dataset("acme/fiqa", split="train", source=source)
    assert report.repository == "acme/fiqa"
    assert report.revision == "abc123def456"
    assert report.license == "mit"
    assert report.split == "train"
    assert report.splits["train"] == 2
    assert "text" in report.candidate_text_columns
    assert report.estimated_rows == 2
    assert (
        report.estimated_characters
        == source.character_stats(
            DatasetRequest(repository="acme/fiqa", split="train"),
            report.candidate_text_columns[0],
        ).estimated_characters
    )
    assert report.to_dict()["revision"] == "abc123def456"


def test_inspect_respects_config_and_split_request() -> None:
    source = FakeDatasetSource(rows=[{"body": "hello"}])
    report = inspect_dataset("org/data", config="sample", split="test", source=source)
    assert report.config == "sample"
    assert report.split == "test"
    source.inspect(DatasetRequest(repository="org/data", config="sample", split="test"))
    assert source.inspect_calls >= 1
