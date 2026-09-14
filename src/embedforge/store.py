"""Local job artifacts under ~/.cache/embedforge/jobs/<id>/."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embedforge.errors import EmbedForgeError
from embedforge.paths import job_dir, jobs_dir
from embedforge.shapes import Job, Plan, Provenance, RowStatus, utc_now

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    timestamp_ms = int(time.time() * 1000)
    if timestamp_ms.bit_length() > 48:
        raise EmbedForgeError("timestamp does not fit ULID")
    randomness = int.from_bytes(os.urandom(10), "big")
    value = (timestamp_ms << 80) | randomness
    chars = ["0"] * 26
    for index in range(25, -1, -1):
        chars[index] = _ULID_ALPHABET[value & 31]
        value >>= 5
    return "".join(chars)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_json(path: Path, data: object) -> None:
    write_text_atomic(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class EmbeddingRecord:
    split: str
    index: int
    status: RowStatus
    cache_key: str | None
    embedding: list[float] | None


def read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise EmbedForgeError(f"expected JSON object in {path}")
    return raw


class JobStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or jobs_dir()

    def directory(self, job_id: str) -> Path:
        return (self.root / job_id) if self.root != jobs_dir() else job_dir(job_id)

    def create(self, job: Job, plan: Plan, provenance: Provenance) -> Path:
        directory = self.directory(job.id)
        if directory.exists():
            raise EmbedForgeError(f"job {job.id} already exists")
        directory.mkdir(parents=True, exist_ok=True)
        self.save_job(job)
        write_json(directory / "plan.json", plan.to_dict())
        (directory / "provenance.yaml").write_text(provenance.to_yaml())
        write_json(directory / "provenance.json", provenance.to_dict())
        (directory / "embeddings.jsonl").touch()
        return directory

    def save_job(self, job: Job) -> None:
        write_json(self.directory(job.id) / "job.json", job.to_dict())

    def load(self, job_id: str) -> Job:
        path = self.directory(job_id) / "job.json"
        if not path.exists():
            raise EmbedForgeError(f"job not found: {job_id}")
        return Job.from_dict(read_json(path))

    def load_plan(self, job_id: str) -> Plan:
        path = self.directory(job_id) / "plan.json"
        if not path.exists():
            raise EmbedForgeError(f"plan not found for job {job_id}")
        return Plan.from_dict(read_json(path))

    def load_provenance(self, job_id: str) -> Provenance:
        path = self.directory(job_id) / "provenance.json"
        if not path.exists():
            raise EmbedForgeError(f"provenance not found for job {job_id}")
        return Provenance.from_dict(read_json(path))

    def output_dir(self, job_id: str) -> Path:
        return self.directory(job_id) / "output"

    def embeddings_path(self, job_id: str) -> Path:
        return self.directory(job_id) / "embeddings.jsonl"

    def append_embedding(
        self,
        job_id: str,
        index: int,
        key: str | None,
        vector: list[float] | None,
        *,
        status: RowStatus,
        split: str,
    ) -> None:
        record = {
            "split": split,
            "index": index,
            "status": status.value,
            "cache_key": key,
            "embedding": vector,
        }
        path = self.embeddings_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _truncate_torn_journal(job_id, path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def load_records(self, job_id: str) -> dict[tuple[str, int], EmbeddingRecord]:
        path = self.embeddings_path(job_id)
        if not path.exists():
            return {}
        loaded: dict[tuple[str, int], EmbeddingRecord] = {}
        complete, _torn = _complete_journal_lines(job_id, path.read_text(encoding="utf-8"))
        for line in complete:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            index = raw.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                continue
            split_raw = raw.get("split", "train")
            split = split_raw if isinstance(split_raw, str) and split_raw else "train"
            status = _record_status(raw)
            embedding = raw.get("embedding")
            vector: list[float] | None
            if embedding is None:
                vector = None
            elif isinstance(embedding, list):
                vector = [float(item) for item in embedding]
            else:
                continue
            key_raw = raw.get("cache_key")
            cache_key = key_raw if isinstance(key_raw, str) else None
            loaded[(split, index)] = EmbeddingRecord(
                split=split,
                index=index,
                status=status,
                cache_key=cache_key,
                embedding=vector,
            )
        return loaded

    def load_embeddings(self, job_id: str) -> dict[int, list[float] | None]:
        loaded: dict[int, list[float] | None] = {}
        for (_split, index), record in self.load_records(job_id).items():
            loaded[index] = record.embedding
        return loaded

    def list_jobs(self) -> list[Job]:
        if not self.root.exists():
            return []
        jobs: list[Job] = []
        for path in sorted(self.root.iterdir()):
            if (path / "job.json").exists():
                jobs.append(Job.from_dict(read_json(path / "job.json")))
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs

    def write_card(self, job_id: str, card: str) -> Path:
        path = self.directory(job_id) / "README.md"
        path.write_text(card)
        return path

    def touch(self, job: Job) -> Job:
        updated = Job(
            id=job.id,
            status=job.status,
            source=job.source,
            embedding=job.embedding,
            output=job.output,
            progress=job.progress,
            created_at=job.created_at,
            updated_at=utc_now(),
            limit=job.limit,
            published_repo=job.published_repo,
            error=job.error,
        )
        self.save_job(updated)
        return updated


def _complete_journal_lines(job_id: str, text: str) -> tuple[list[str], bool]:
    lines = [line for line in text.splitlines() if line.strip()]
    complete: list[str] = []
    for offset, line in enumerate(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError:
            if offset == len(lines) - 1:
                return complete, True
            raise EmbedForgeError(f"corrupt embeddings journal in job {job_id}") from None
        complete.append(line)
    return complete, False


def _truncate_torn_journal(job_id: str, path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    complete, torn = _complete_journal_lines(job_id, text)
    normalized = "".join(f"{line}\n" for line in complete)
    if torn or text != normalized:
        write_text_atomic(path, normalized)


def _record_status(raw: dict[str, Any]) -> RowStatus:
    status_raw = raw.get("status")
    if isinstance(status_raw, str):
        try:
            return RowStatus(status_raw)
        except ValueError:
            pass
    if isinstance(raw.get("embedding"), list):
        return RowStatus.SUCCESS
    return RowStatus.FAILED
