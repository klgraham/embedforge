from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from embedforge.hfdata import CharStats, DatasetRequest, LoadedDataset
from embedforge.providers import EmbedBatch
from embedforge.shapes import ColumnInfo, InspectReport, SourceRef


@dataclass
class FakeDatasetSource:
    rows: Sequence[Mapping[str, object]]
    repository: str = "acme/fiqa"
    revision: str = "abc123def456"
    license: str | None = "mit"
    config: str = "default"
    split: str = "train"
    inspect_calls: int = 0
    load_calls: int = 0

    def inspect(self, request: DatasetRequest) -> InspectReport:
        self.inspect_calls += 1
        columns = tuple(
            ColumnInfo(name=name, dtype="string")
            for name in (self.rows[0].keys() if self.rows else ["text"])
        )
        candidates = tuple(col.name for col in columns if col.dtype == "string")
        stats = self.character_stats(request, candidates[0] if candidates else "text")
        return InspectReport(
            repository=request.repository or self.repository,
            revision=self.revision,
            license=self.license,
            config=request.config or self.config,
            split=request.split or self.split,
            configs=(self.config,),
            splits={self.split: len(self.rows)},
            columns=columns,
            candidate_text_columns=candidates,
            estimated_rows=len(self.rows),
            estimated_characters=stats.estimated_characters,
        )

    def character_stats(self, request: DatasetRequest, column: str) -> CharStats:
        characters = 0
        for row in self.rows:
            value = row.get(column)
            characters += len(value) if isinstance(value, str) else 0
        return CharStats(
            rows=len(self.rows),
            estimated_characters=characters,
            sampled_rows=len(self.rows),
        )

    def load_rows(self, request: DatasetRequest, *, limit: int | None) -> LoadedDataset:
        self.load_calls += 1
        selected = self.rows[:limit] if limit is not None else self.rows
        rows = [dict(row) for row in selected]
        columns = tuple(self.rows[0].keys()) if self.rows else ()
        return LoadedDataset(
            rows=rows,
            columns=columns,
            source=SourceRef(
                repository=request.repository,
                revision=self.revision,
                config=request.config or self.config,
                split=request.split or self.split,
                license=self.license,
            ),
        )


@dataclass
class FakeEmbedder:
    dimensions: int = 4
    calls: list[list[str]] = field(default_factory=list)
    fail_after: int | None = None

    def embed(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> EmbedBatch:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("injected embedder failure")
        self.calls.append(list(texts))
        dim = dimensions or self.dimensions
        vectors = [_vector(text, dim) for text in texts]
        return EmbedBatch(vectors=vectors, tokens=sum(max(1, len(text) // 4) for text in texts))

    @property
    def embedded_texts(self) -> list[str]:
        return [text for batch in self.calls for text in batch]


@dataclass
class FakePublisher:
    calls: list[dict[str, object]] = field(default_factory=list)

    def publish(
        self,
        *,
        repo_id: str,
        private: bool,
        revision: str | None,
        dataset_dir: object,
        card: str,
        provenance_yaml: str,
        token: str | None,
    ) -> str:
        self.calls.append(
            {
                "repo_id": repo_id,
                "private": private,
                "revision": revision,
                "dataset_dir": str(dataset_dir),
                "card": card,
                "provenance_yaml": provenance_yaml,
                "token": token,
            }
        )
        return f"https://huggingface.co/datasets/{repo_id}"


def _vector(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [((digest[index % 32] / 127.5) - 1.0) for index in range(dimensions)]
