from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from datasets import DatasetDict, load_from_disk

from embedforge.hfdata import CharStats, DatasetRequest, LoadedDataset, LoadedSplit
from embedforge.providers import EmbedBatch
from embedforge.publish import DestinationInfo
from embedforge.shapes import ColumnInfo, InspectReport, SourceRef


@dataclass
class FakeDatasetSource:
    rows: Sequence[Mapping[str, object]] = ()
    repository: str = "acme/fiqa"
    revision: str = "abc123def456"
    license: str | None = "mit"
    config: str = "default"
    split: str = "train"
    split_data: dict[str, Sequence[Mapping[str, object]]] | None = None
    inspect_calls: int = 0
    load_calls: int = 0

    def _all_splits(self) -> dict[str, list[dict[str, object]]]:
        if self.split_data is not None:
            return {name: [dict(row) for row in rows] for name, rows in self.split_data.items()}
        return {self.split: [dict(row) for row in self.rows]}

    def inspect(self, request: DatasetRequest) -> InspectReport:
        self.inspect_calls += 1
        splits = self._all_splits()
        sample_rows = splits.get(request.split or self.split) or next(iter(splits.values()), [])
        columns = tuple(
            ColumnInfo(name=name, dtype="string")
            for name in (sample_rows[0].keys() if sample_rows else ["text"])
        )
        candidates = tuple(col.name for col in columns if col.dtype == "string")
        stats = self.character_stats(
            DatasetRequest(
                repository=request.repository,
                config=request.config,
                split=request.split or self.split,
            ),
            candidates[0] if candidates else "text",
        )
        return InspectReport(
            repository=request.repository or self.repository,
            revision=self.revision,
            license=self.license,
            config=request.config or self.config,
            split=request.split or self.split,
            configs=(self.config,),
            splits={name: len(rows) for name, rows in splits.items()},
            columns=columns,
            candidate_text_columns=candidates,
            estimated_rows=stats.rows,
            estimated_characters=stats.estimated_characters,
        )

    def character_stats(self, request: DatasetRequest, column: str) -> CharStats:
        splits = self._all_splits()
        names = [request.split] if request.split else list(splits)
        rows: list[dict[str, object]] = []
        for name in names:
            if name in splits:
                rows.extend(splits[name])
        characters = 0
        for row in rows:
            value = row.get(column)
            characters += len(value) if isinstance(value, str) else 0
        return CharStats(
            rows=len(rows),
            estimated_characters=characters,
            sampled_rows=len(rows),
        )

    def load_rows(self, request: DatasetRequest, *, limit: int | None) -> LoadedDataset:
        self.load_calls += 1
        splits = self._all_splits()
        names = [request.split] if request.split is not None else list(splits)
        loaded: list[LoadedSplit] = []
        for name in names:
            rows = splits.get(name, [])
            selected = rows[:limit] if limit is not None else rows
            columns = tuple(selected[0].keys()) if selected else ()
            loaded.append(LoadedSplit(name=name, rows=selected, columns=columns))
        return LoadedDataset(
            splits=tuple(loaded),
            source=SourceRef(
                repository=request.repository,
                revision=self.revision,
                config=request.config or self.config,
                split=request.split,
                license=self.license,
            ),
        )


@dataclass
class FakeEmbedder:
    dimensions: int = 4
    calls: list[list[str]] = field(default_factory=list)
    fail_after: int | None = None
    fail_on: set[str] = field(default_factory=set)
    seen_dimensions: list[int | None] = field(default_factory=list)

    def embed(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> EmbedBatch:
        self.seen_dimensions.append(dimensions)
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("injected embedder failure")
        if any(text in self.fail_on for text in texts):
            raise RuntimeError("injected provider failure")
        self.calls.append(list(texts))
        dim = dimensions or self.dimensions
        vectors = [_vector(text, dim) for text in texts]
        return EmbedBatch(vectors=vectors, tokens=sum(max(1, len(text) // 4) for text in texts))

    @property
    def embedded_texts(self) -> list[str]:
        return [text for batch in self.calls for text in batch]


@dataclass
class FakeHub:
    """Models Hub SDK behavior: private=True does not flip an existing public repo."""

    repos: dict[str, dict[str, object]] = field(default_factory=dict)
    uploads: list[dict[str, object]] = field(default_factory=list)
    files: dict[tuple[str, str, str | None], str] = field(default_factory=dict)

    def inspect_destination(self, repo_id: str, token: str | None) -> DestinationInfo:
        del token
        if repo_id not in self.repos:
            return DestinationInfo(exists=False, private=None)
        private = self.repos[repo_id].get("private")
        return DestinationInfo(exists=True, private=bool(private))

    def push_dataset(
        self,
        dataset_dir: Path,
        repo_id: str,
        *,
        private: bool,
        revision: str | None,
        config_name: str | None,
        token: str | None,
    ) -> None:
        del token
        staged = load_from_disk(str(dataset_dir))
        if isinstance(staged, DatasetDict):
            split_names = list(staged.keys())
        else:
            split_names = []
        if repo_id not in self.repos:
            self.repos[repo_id] = {"private": private}
        self.files[(repo_id, "README.md", revision)] = hub_generated_readme(
            config_name or "default",
            split_names or ["train"],
        )
        self.uploads.append(
            {
                "kind": "dataset",
                "repo_id": repo_id,
                "private_flag": private,
                "revision": revision,
                "config_name": config_name,
                "splits": split_names,
                "actual_private": self.repos[repo_id]["private"],
            }
        )

    def read_text(
        self,
        path_in_repo: str,
        repo_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> str:
        del token
        key = (repo_id, path_in_repo, revision)
        if key not in self.files:
            raise FileNotFoundError(f"{path_in_repo} missing from {repo_id}")
        return self.files[key]

    def upload_text(
        self,
        content: str,
        path_in_repo: str,
        repo_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> None:
        del token
        self.files[(repo_id, path_in_repo, revision)] = content
        self.uploads.append(
            {
                "kind": path_in_repo,
                "repo_id": repo_id,
                "revision": revision,
                "content": content,
            }
        )


def hub_generated_readme(config_name: str, splits: Sequence[str]) -> str:
    """Hub-style card with the pushed config plus an existing non-default mapping."""
    pushed_files = "\n".join(
        f"  - split: {name}\n    path: data/{name}-*" for name in splits
    )
    return (
        "---\n"
        "configs:\n"
        f"- config_name: {config_name}\n"
        "  data_files:\n"
        f"{pushed_files}\n"
        "- config_name: corpus\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: corpus/train-*\n"
        "dataset_info:\n"
        f"- config_name: {config_name}\n"
        "  splits:\n"
        f"  - name: {splits[0]}\n"
        "    num_examples: 1\n"
        "- config_name: corpus\n"
        "  splits:\n"
        "  - name: train\n"
        "    num_examples: 10\n"
        "---\n\n"
        "# Auto-generated dataset card\n"
    )


def _vector(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [((digest[index % 32] / 127.5) - 1.0) for index in range(dimensions)]
