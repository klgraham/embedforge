"""Local job artifacts under ~/.cache/embedforge/jobs/<id>/."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from embedforge.errors import EmbedForgeError
from embedforge.paths import job_dir, jobs_dir
from embedforge.shapes import Job, Plan, Provenance, utc_now

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


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


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
        self, job_id: str, index: int, key: str | None, vector: list[float] | None
    ) -> None:
        record = {"index": index, "cache_key": key, "embedding": vector}
        path = self.embeddings_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def load_embeddings(self, job_id: str) -> dict[int, list[float] | None]:
        path = self.embeddings_path(job_id)
        if not path.exists():
            return {}
        loaded: dict[int, list[float] | None] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            index = raw.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                continue
            embedding = raw.get("embedding")
            if embedding is None:
                loaded[index] = None
            elif isinstance(embedding, list):
                loaded[index] = [float(item) for item in embedding]
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
