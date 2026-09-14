"""Named V1 data shapes: config, job, plan, and provenance."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from embedforge.errors import EmbedForgeError

ProviderName = Literal["openai", "openrouter"]

ALLOWED_CONFIG_KEYS = (
    "provider",
    "model",
    "batch_size",
    "concurrency",
    "dimensions",
    "output_column",
    "hf_namespace",
)

DEFAULT_PROVIDER: ProviderName = "openai"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 128
DEFAULT_CONCURRENCY = 8
DEFAULT_OUTPUT_COLUMN = "embedding"


class Provider(StrEnum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"


class JobStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    PUBLISHED = "published"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_provider(value: str) -> Provider:
    normalized = value.strip().lower()
    for provider in Provider:
        if provider.value == normalized:
            return provider
    allowed = ", ".join(p.value for p in Provider)
    raise EmbedForgeError(f"unknown provider {value!r}; expected one of: {allowed}")


@dataclass(frozen=True)
class Config:
    """Resolved user settings. Secrets are never stored here."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    concurrency: int = DEFAULT_CONCURRENCY
    dimensions: int | None = None
    output_column: str = DEFAULT_OUTPUT_COLUMN
    hf_namespace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "dimensions": self.dimensions,
            "output_column": self.output_column,
            "hf_namespace": self.hf_namespace,
        }


@dataclass(frozen=True)
class SourceRef:
    repository: str
    revision: str
    config: str | None
    split: str | None
    license: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "config": self.config,
            "split": self.split,
            "license": self.license,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SourceRef:
        return SourceRef(
            repository=_require_str(data, "repository"),
            revision=_require_str(data, "revision"),
            config=_optional_str(data.get("config")),
            split=_optional_str(data.get("split")),
            license=_optional_str(data.get("license")),
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int
    column: str
    source_columns: tuple[str, ...]
    batch_size: int
    concurrency: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "column": self.column,
            "source_columns": list(self.source_columns),
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> EmbeddingSettings:
        raw_columns = data.get("source_columns", [])
        if not isinstance(raw_columns, list) or not all(
            isinstance(item, str) for item in raw_columns
        ):
            raise EmbedForgeError("embedding.source_columns must be a list of strings")
        return EmbeddingSettings(
            provider=_require_str(data, "provider"),
            model=_require_str(data, "model"),
            dimensions=_require_int(data, "dimensions"),
            column=_require_str(data, "column"),
            source_columns=tuple(raw_columns),
            batch_size=_require_int(data, "batch_size"),
            concurrency=_require_int(data, "concurrency"),
        )


@dataclass(frozen=True)
class OutputSpec:
    repo: str
    column: str

    def to_dict(self) -> dict[str, Any]:
        return {"repo": self.repo, "column": self.column}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> OutputSpec:
        return OutputSpec(
            repo=_require_str(data, "repo"),
            column=_require_str(data, "column"),
        )


@dataclass(frozen=True)
class Estimates:
    rows: int
    characters: int
    tokens: int
    cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "characters": self.characters,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Estimates:
        return Estimates(
            rows=_require_int(data, "rows"),
            characters=_require_int(data, "characters"),
            tokens=_require_int(data, "tokens"),
            cost_usd=_require_float(data, "cost_usd"),
        )


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype}


@dataclass(frozen=True)
class InspectReport:
    repository: str
    revision: str
    license: str | None
    config: str | None
    split: str | None
    configs: tuple[str, ...]
    splits: dict[str, int]
    columns: tuple[ColumnInfo, ...]
    candidate_text_columns: tuple[str, ...]
    estimated_rows: int
    estimated_characters: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "license": self.license,
            "config": self.config,
            "split": self.split,
            "configs": list(self.configs),
            "splits": dict(self.splits),
            "columns": [column.to_dict() for column in self.columns],
            "candidate_text_columns": list(self.candidate_text_columns),
            "estimated_rows": self.estimated_rows,
            "estimated_characters": self.estimated_characters,
        }


@dataclass(frozen=True)
class Plan:
    """First-class paid-operation gate. Building a plan never embeds."""

    source: SourceRef
    embedding: EmbeddingSettings
    output: OutputSpec
    estimates: Estimates
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "embedding": self.embedding.to_dict(),
            "output": self.output.to_dict(),
            "estimates": self.estimates.to_dict(),
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Plan:
        return Plan(
            source=SourceRef.from_dict(_require_object(data, "source")),
            embedding=EmbeddingSettings.from_dict(_require_object(data, "embedding")),
            output=OutputSpec.from_dict(_require_object(data, "output")),
            estimates=Estimates.from_dict(_require_object(data, "estimates")),
            created_at=_require_str(data, "created_at"),
        )


@dataclass(frozen=True)
class JobProgress:
    embedded: int = 0
    skipped: int = 0
    failed: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    next_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedded": self.embedded,
            "skipped": self.skipped,
            "failed": self.failed,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "next_index": self.next_index,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> JobProgress:
        return JobProgress(
            embedded=_require_int(data, "embedded"),
            skipped=_require_int(data, "skipped"),
            failed=_require_int(data, "failed"),
            tokens=_require_int(data, "tokens"),
            cost_usd=_require_float(data, "cost_usd"),
            next_index=_require_int(data, "next_index"),
        )


@dataclass(frozen=True)
class Job:
    id: str
    status: JobStatus
    source: SourceRef
    embedding: EmbeddingSettings
    output: OutputSpec
    progress: JobProgress
    created_at: str
    updated_at: str
    limit: int | None = None
    published_repo: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "source": self.source.to_dict(),
            "embedding": self.embedding.to_dict(),
            "output": self.output.to_dict(),
            "progress": self.progress.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "limit": self.limit,
            "published_repo": self.published_repo,
            "error": self.error,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Job:
        status_raw = _require_str(data, "status")
        try:
            status = JobStatus(status_raw)
        except ValueError as exc:
            raise EmbedForgeError(f"unknown job status {status_raw!r}") from exc
        return Job(
            id=_require_str(data, "id"),
            status=status,
            source=SourceRef.from_dict(_require_object(data, "source")),
            embedding=EmbeddingSettings.from_dict(_require_object(data, "embedding")),
            output=OutputSpec.from_dict(_require_object(data, "output")),
            progress=JobProgress.from_dict(_require_object(data, "progress")),
            created_at=_require_str(data, "created_at"),
            updated_at=_require_str(data, "updated_at"),
            limit=_optional_int(data.get("limit")),
            published_repo=_optional_str(data.get("published_repo")),
            error=_optional_str(data.get("error")),
        )


@dataclass(frozen=True)
class Provenance:
    """Machine-readable metadata written onto every generated dataset."""

    embedforge_version: str
    source: SourceRef
    embedding: EmbeddingSettings
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedforge": {"version": self.embedforge_version},
            "source": {
                "repository": self.source.repository,
                "revision": self.source.revision,
                "config": self.source.config,
                "split": self.source.split,
                "license": self.source.license,
            },
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "dimensions": self.embedding.dimensions,
                "column": self.embedding.column,
                "source_columns": list(self.embedding.source_columns),
                "created_at": self.created_at,
            },
        }

    def to_yaml(self) -> str:
        return dump_yaml(self.to_dict())

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Provenance:
        embedforge = _require_object(data, "embedforge")
        embedding_raw = _require_object(data, "embedding")
        source = SourceRef.from_dict(_require_object(data, "source"))
        source_columns = embedding_raw.get("source_columns", [])
        if not isinstance(source_columns, list) or not all(
            isinstance(item, str) for item in source_columns
        ):
            raise EmbedForgeError("provenance embedding.source_columns must be strings")
        settings = EmbeddingSettings(
            provider=_require_str(embedding_raw, "provider"),
            model=_require_str(embedding_raw, "model"),
            dimensions=_require_int(embedding_raw, "dimensions"),
            column=_require_str(embedding_raw, "column"),
            source_columns=tuple(source_columns),
            batch_size=1,
            concurrency=1,
        )
        return Provenance(
            embedforge_version=_require_str(embedforge, "version"),
            source=source,
            embedding=settings,
            created_at=_require_str(embedding_raw, "created_at"),
        )


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "message": self.message}


@dataclass(frozen=True)
class Diagnostic:
    name: str
    message: str
    value: float | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "message": self.message, "value": self.value}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    checks: tuple[Check, ...]
    diagnostics: tuple[Diagnostic, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class PublishResult:
    repo_id: str
    private: bool
    revision: str | None
    url: str
    card: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "private": self.private,
            "revision": self.revision,
            "url": self.url,
            "card": self.card,
        }


def dump_yaml(value: object, *, indent: int = 0) -> str:
    """Minimal YAML emitter for provenance and cards (no extra dependency)."""
    prefix = "  " * indent
    if isinstance(value, dict):
        if not value:
            return "{}\n"
        lines: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise EmbedForgeError("yaml keys must be strings")
            if _is_scalar(item):
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
            elif isinstance(item, list):
                if not item:
                    lines.append(f"{prefix}{key}: []")
                elif all(_is_scalar(child) for child in item):
                    lines.append(f"{prefix}{key}:")
                    for child in item:
                        lines.append(f"{prefix}  - {_yaml_scalar(child)}")
                else:
                    lines.append(f"{prefix}{key}:")
                    lines.append(dump_yaml(item, indent=indent + 1).rstrip("\n"))
            elif isinstance(item, dict):
                lines.append(f"{prefix}{key}:")
                dumped = dump_yaml(item, indent=indent + 1).rstrip("\n")
                lines.append(dumped if dumped else f"{prefix}  {{}}")
            else:
                raise EmbedForgeError(f"cannot encode yaml value: {type(item).__name__}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]\n"
        lines = []
        for item in value:
            if _is_scalar(item):
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
            else:
                nested = dump_yaml(item, indent=indent + 1).rstrip("\n")
                lines.append(f"{prefix}-")
                lines.append(nested)
        return "\n".join(lines) + "\n"
    return f"{prefix}{_yaml_scalar(value)}\n"


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        if value == "" or _needs_yaml_quote(value):
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        return value
    raise EmbedForgeError(f"cannot encode yaml scalar: {type(value).__name__}")


def _needs_yaml_quote(value: str) -> bool:
    if value.lower() in {"null", "true", "false", "yes", "no"}:
        return True
    return any(ch in value for ch in ":#{}[],&*!|>%@`'\"\n")


def _require_object(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise EmbedForgeError(f"expected object for {key}")
    return value


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise EmbedForgeError(f"expected non-empty string for {key}")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EmbedForgeError(f"expected integer for {key}")
    return value


def _require_float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EmbedForgeError(f"expected number for {key}")
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise EmbedForgeError("expected string or null")
    return value or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise EmbedForgeError("expected integer or null")
    return value


def replace_job(job: Job, **changes: Any) -> Job:
    values = {field.name: getattr(job, field.name) for field in fields(job)}
    values.update(changes)
    values["updated_at"] = utc_now()
    return Job(**values)
