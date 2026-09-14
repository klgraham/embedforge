"""Local job artifacts under ~/.cache/embedforge/jobs/<id>/."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

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
        self._writers: dict[str, IO[str]] = {}

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

    def open_journal(self, job_id: str) -> dict[tuple[str, int], EmbeddingRecord]:
        """Repair the journal once and keep an append handle for later writes."""
        existing = self._writers.get(job_id)
        if existing is not None:
            existing.flush()
            return self.load_records(job_id)
        path = self.embeddings_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        loaded: dict[tuple[str, int], EmbeddingRecord] = {}
        if path.exists():
            text = path.read_text(encoding="utf-8")
            complete, loaded, torn = _parse_journal(job_id, text)
            normalized = "".join(f"{line}\n" for line in complete)
            if torn or text != normalized:
                write_text_atomic(path, normalized)
        else:
            path.touch()
        self._writers[job_id] = path.open("a", encoding="utf-8")
        return loaded

    def close_journal(self, job_id: str) -> None:
        handle = self._writers.pop(job_id, None)
        if handle is not None:
            handle.close()

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
        handle = self._writers.get(job_id)
        if handle is None:
            self.open_journal(job_id)
            handle = self._writers[job_id]
        handle.write(json.dumps(record) + "\n")
        handle.flush()

    def load_records(self, job_id: str) -> dict[tuple[str, int], EmbeddingRecord]:
        handle = self._writers.get(job_id)
        if handle is not None:
            handle.flush()
        path = self.embeddings_path(job_id)
        if not path.exists():
            return {}
        _complete, loaded, _torn = _parse_journal(job_id, path.read_text(encoding="utf-8"))
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


def _journal_loads(line: str) -> Any:
    return json.loads(line)


def _parse_journal(
    job_id: str, text: str
) -> tuple[list[str], dict[tuple[str, int], EmbeddingRecord], bool]:
    lines = [line for line in text.splitlines() if line.strip()]
    complete: list[str] = []
    loaded: dict[tuple[str, int], EmbeddingRecord] = {}
    for offset, line in enumerate(lines):
        try:
            raw = _journal_loads(line)
        except json.JSONDecodeError:
            if offset == len(lines) - 1:
                return complete, loaded, True
            raise EmbedForgeError(f"corrupt embeddings journal in job {job_id}") from None
        complete.append(line)
        record = _record_from_raw(raw)
        if record is not None:
            loaded[(record.split, record.index)] = record
    return complete, loaded, False


def _record_from_raw(raw: object) -> EmbeddingRecord | None:
    if not isinstance(raw, dict):
        return None
    index = raw.get("index")
    if not isinstance(index, int) or isinstance(index, bool):
        return None
    split_raw = raw.get("split", "train")
    split = split_raw if isinstance(split_raw, str) and split_raw else "train"
    embedding = raw.get("embedding")
    vector: list[float] | None
    if embedding is None:
        vector = None
    elif isinstance(embedding, list):
        vector = [float(item) for item in embedding]
    else:
        return None
    key_raw = raw.get("cache_key")
    cache_key = key_raw if isinstance(key_raw, str) else None
    return EmbeddingRecord(
        split=split,
        index=index,
        status=_record_status(raw),
        cache_key=cache_key,
        embedding=vector,
    )


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
