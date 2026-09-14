"""Run and resume embedding jobs, writing local artifacts."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Never

from datasets import Dataset, DatasetDict

from embedforge import __version__
from embedforge.cache import EmbeddingCache, cache_key
from embedforge.catalog import api_dimensions, estimate_cost_usd, resolve_model
from embedforge.config import apply_overrides, load_config
from embedforge.errors import EmbedForgeError
from embedforge.hfdata import DatasetRequest, DatasetSource, HuggingFaceDatasetSource, LoadedDataset
from embedforge.plan import build_plan
from embedforge.providers import EmbedBatch, Embedder, get_embedder
from embedforge.shapes import (
    Config,
    Job,
    JobProgress,
    JobStatus,
    Plan,
    Provenance,
    RowStatus,
    replace_job,
    utc_now,
)
from embedforge.store import EmbeddingRecord, JobStore, new_ulid

ProgressCallback = Callable[[str], None]
WorkItem = tuple[str, int, str]


@dataclass
class RunResult:
    job: Job
    output_path: str


def start_or_resume(
    repository: str | None,
    *,
    column: str | None = None,
    resume_id: str | None = None,
    config: Config | None = None,
    provider: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    dataset_config: str | None = None,
    split: str | None = None,
    output_column: str | None = None,
    batch_size: int | None = None,
    concurrency: int | None = None,
    limit: int | None = None,
    source: DatasetSource | None = None,
    store: JobStore | None = None,
    cache: EmbeddingCache | None = None,
    embedder: Embedder | None = None,
    embedder_factory: Callable[[str], Embedder] | None = None,
    on_progress: ProgressCallback | None = None,
) -> RunResult:
    job_store = store or JobStore()
    embedding_cache = cache or EmbeddingCache()
    gateway = source or HuggingFaceDatasetSource()
    resolved = apply_overrides(
        config or load_config(),
        provider=provider,
        model=model,
        dimensions=dimensions,
        output_column=output_column,
        batch_size=batch_size,
        concurrency=concurrency,
    )
    if resume_id:
        job = job_store.load(resume_id)
        if job.status is JobStatus.PUBLISHED:
            raise EmbedForgeError(f"job {job.id} is already published")
        if job.status is JobStatus.COMPLETED and job.progress.failed == 0:
            output = job_store.output_dir(job.id)
            if output.exists():
                return RunResult(job=job, output_path=str(output))
    else:
        if not repository or not column:
            raise EmbedForgeError("dataset and --column are required unless resuming")
        plan = build_plan(
            repository,
            column=column,
            config=resolved,
            provider=resolved.provider,
            model=resolved.model,
            dimensions=resolved.dimensions,
            dataset_config=dataset_config,
            split=split,
            output_column=resolved.output_column,
            limit=limit,
            source=gateway,
        )
        if batch_size is not None:
            plan = _plan_with_batch(plan, batch_size, resolved.concurrency)
        if concurrency is not None:
            plan = _plan_with_batch(plan, plan.embedding.batch_size, concurrency)
        created = utc_now()
        job = Job(
            id=new_ulid(),
            status=JobStatus.CREATED,
            source=plan.source,
            embedding=plan.embedding,
            output=plan.output,
            progress=JobProgress(),
            created_at=created,
            updated_at=created,
            limit=limit,
        )
        provenance = Provenance(
            embedforge_version=__version__,
            source=plan.source,
            embedding=plan.embedding,
            created_at=created,
        )
        job_store.create(job, plan, provenance)

    worker = embedder
    if worker is None:
        factory = embedder_factory or get_embedder
        worker = factory(job.embedding.provider)
    return execute_job(
        job,
        store=job_store,
        cache=embedding_cache,
        source=gateway,
        embedder=worker,
        on_progress=on_progress,
    )


def resume_job(
    job_id: str,
    *,
    source: DatasetSource | None = None,
    store: JobStore | None = None,
    cache: EmbeddingCache | None = None,
    embedder: Embedder | None = None,
    embedder_factory: Callable[[str], Embedder] | None = None,
    on_progress: ProgressCallback | None = None,
) -> RunResult:
    return start_or_resume(
        None,
        resume_id=job_id,
        source=source,
        store=store,
        cache=cache,
        embedder=embedder,
        embedder_factory=embedder_factory,
        on_progress=on_progress,
    )


def execute_job(
    job: Job,
    *,
    store: JobStore,
    cache: EmbeddingCache,
    source: DatasetSource,
    embedder: Embedder,
    on_progress: ProgressCallback | None = None,
) -> RunResult:
    loaded = source.load_rows(
        DatasetRequest(
            repository=job.source.repository,
            config=job.source.config,
            split=job.source.split,
            revision=job.source.revision,
        ),
        limit=job.limit,
    )
    if job.embedding.column in loaded.columns:
        raise EmbedForgeError(
            f"output column {job.embedding.column!r} collides with a source column"
        )
    items = _work_items(loaded, job.embedding.source_columns[0])
    started = time.monotonic()
    job = replace_job(job, status=JobStatus.RUNNING, error=None)
    store.save_job(job)
    request_dims = api_dimensions(job.embedding.model, job.embedding.dimensions)
    model_info = resolve_model(
        job.embedding.provider, job.embedding.model, job.embedding.dimensions
    )

    try:
        records = store.open_journal(job.id)
        pending = _pending_items(items, records, job, store, cache)
        session_embedded = 0
        tokens = job.progress.tokens
        cost = job.progress.cost_usd
        indexed = _index_pending(pending, job)
        windows = _chunk(indexed.unique, job.embedding.batch_size)
        for window in _group_windows(windows, job.embedding.concurrency):
            texts_by_chunk = [[text for _key, text in chunk] for chunk in window]
            batch_results = _embed_chunks(texts_by_chunk, job, embedder, request_dims)
            for chunk, result in zip(window, batch_results, strict=True):
                batch, batch_error = result
                if batch_error is not None or batch is None:
                    for key, text in chunk:
                        try:
                            single = embedder.embed(
                                [text],
                                model=job.embedding.model,
                                dimensions=request_dims,
                            )
                        except Exception:
                            _persist_group(
                                indexed.rows_by_key[key],
                                job,
                                store,
                                records,
                                status=RowStatus.FAILED,
                                vector=None,
                                cache_key_value=None,
                            )
                            continue
                        tokens += single.tokens
                        cost += estimate_cost_usd(single.tokens, model_info.usd_per_million_tokens)
                        session_embedded += _store_unique_success(
                            job,
                            cache,
                            store,
                            records,
                            key,
                            text,
                            single.vectors[0],
                            indexed.rows_by_key[key],
                        )
                    continue
                tokens += batch.tokens
                cost += estimate_cost_usd(batch.tokens, model_info.usd_per_million_tokens)
                for (key, text), vector in zip(chunk, batch.vectors, strict=True):
                    session_embedded += _store_unique_success(
                        job,
                        cache,
                        store,
                        records,
                        key,
                        text,
                        vector,
                        indexed.rows_by_key[key],
                    )
            progress = _progress_from_records(
                items,
                records,
                session_embedded=session_embedded,
                tokens=tokens,
                cost_usd=cost,
            )
            job = replace_job(job, status=JobStatus.RUNNING, progress=progress)
            store.save_job(job)
            if on_progress:
                on_progress(_progress_line(progress, time.monotonic() - started))

        progress = _progress_from_records(
            items,
            records,
            session_embedded=session_embedded,
            tokens=tokens,
            cost_usd=cost,
        )
        output_path = _materialize_dataset(job, loaded, store, records)
        status = JobStatus.COMPLETED if progress.failed == 0 else JobStatus.FAILED
        job = replace_job(
            job,
            status=status,
            progress=progress,
            error=None if progress.failed == 0 else f"{progress.failed} rows failed",
        )
        store.save_job(job)
        if on_progress:
            on_progress(_progress_line(progress, time.monotonic() - started))
            on_progress(f"output: {output_path}")
        return RunResult(job=job, output_path=str(output_path))
    except Exception as exc:
        job = replace_job(job, status=JobStatus.FAILED, error=str(exc))
        store.save_job(job)
        if isinstance(exc, EmbedForgeError):
            raise
        raise EmbedForgeError(str(exc)) from exc
    finally:
        store.close_journal(job.id)


def _work_items(loaded: LoadedDataset, column: str) -> list[WorkItem]:
    items: list[WorkItem] = []
    for split in loaded.splits:
        for index, row in enumerate(split.rows):
            items.append((split.name, index, _as_text(row.get(column))))
    return items


def _is_resolved(record: EmbeddingRecord | None) -> bool:
    if record is None:
        return False
    match record.status:
        case RowStatus.SUCCESS | RowStatus.SKIPPED:
            return True
        case RowStatus.FAILED:
            return False
        case _:
            never: Never = record.status
            raise EmbedForgeError(f"unknown row status: {never}")


def _pending_items(
    items: list[WorkItem],
    records: dict[tuple[str, int], EmbeddingRecord],
    job: Job,
    store: JobStore,
    cache: EmbeddingCache,
) -> list[WorkItem]:
    pending: list[WorkItem] = []
    for split, index, text in items:
        record = records.get((split, index))
        if _is_resolved(record):
            continue
        if text == "":
            store.append_embedding(job.id, index, None, None, status=RowStatus.SKIPPED, split=split)
            records[(split, index)] = EmbeddingRecord(
                split=split,
                index=index,
                status=RowStatus.SKIPPED,
                cache_key=None,
                embedding=None,
            )
            continue
        cached = cache.get(
            job.embedding.provider,
            job.embedding.model,
            job.embedding.dimensions,
            text,
        )
        if cached is not None:
            key = _key(job, text)
            store.append_embedding(
                job.id, index, key, cached, status=RowStatus.SUCCESS, split=split
            )
            records[(split, index)] = EmbeddingRecord(
                split=split,
                index=index,
                status=RowStatus.SUCCESS,
                cache_key=key,
                embedding=cached,
            )
            continue
        pending.append((split, index, text))
    return pending


@dataclass(frozen=True)
class PendingIndex:
    unique: list[tuple[str, str]]
    rows_by_key: dict[str, list[WorkItem]]


def _index_pending(pending: list[WorkItem], job: Job) -> PendingIndex:
    """Hash each pending row once and group duplicates by cache key."""
    rows_by_key: dict[str, list[WorkItem]] = {}
    unique: list[tuple[str, str]] = []
    for item in pending:
        key = _key(job, item[2])
        group = rows_by_key.get(key)
        if group is None:
            rows_by_key[key] = [item]
            unique.append((key, item[2]))
        else:
            group.append(item)
    return PendingIndex(unique=unique, rows_by_key=rows_by_key)


def _store_unique_success(
    job: Job,
    cache: EmbeddingCache,
    store: JobStore,
    records: dict[tuple[str, int], EmbeddingRecord],
    key: str,
    text: str,
    vector: list[float],
    rows: list[WorkItem],
) -> int:
    cache.put(
        job.embedding.provider,
        job.embedding.model,
        job.embedding.dimensions,
        text,
        vector,
    )
    for split, index, _text in rows:
        store.append_embedding(job.id, index, key, vector, status=RowStatus.SUCCESS, split=split)
        records[(split, index)] = EmbeddingRecord(
            split=split,
            index=index,
            status=RowStatus.SUCCESS,
            cache_key=key,
            embedding=vector,
        )
    return len(rows)


def _persist_group(
    rows: list[WorkItem],
    job: Job,
    store: JobStore,
    records: dict[tuple[str, int], EmbeddingRecord],
    *,
    status: RowStatus,
    vector: list[float] | None,
    cache_key_value: str | None,
) -> None:
    for split, index, _text in rows:
        store.append_embedding(job.id, index, cache_key_value, vector, status=status, split=split)
        records[(split, index)] = EmbeddingRecord(
            split=split,
            index=index,
            status=status,
            cache_key=cache_key_value,
            embedding=vector,
        )


def _progress_from_records(
    items: list[WorkItem],
    records: dict[tuple[str, int], EmbeddingRecord],
    *,
    session_embedded: int,
    tokens: int,
    cost_usd: float,
) -> JobProgress:
    success = 0
    skipped_empty = 0
    failed = 0
    for split, index, _text in items:
        record = records.get((split, index))
        if record is None:
            continue
        match record.status:
            case RowStatus.SUCCESS:
                success += 1
            case RowStatus.SKIPPED:
                skipped_empty += 1
            case RowStatus.FAILED:
                failed += 1
            case _:
                never: Never = record.status
                raise EmbedForgeError(f"unknown row status: {never}")
    prior_or_cache = max(0, success - session_embedded)
    return JobProgress(
        embedded=session_embedded,
        skipped=skipped_empty + prior_or_cache,
        failed=failed,
        tokens=tokens,
        cost_usd=cost_usd,
        next_index=success + skipped_empty,
    )


def _embed_chunks(
    chunks: list[list[str]],
    job: Job,
    embedder: Embedder,
    request_dims: int | None,
) -> list[tuple[EmbedBatch | None, Exception | None]]:
    if job.embedding.concurrency <= 1 or len(chunks) <= 1:
        results: list[tuple[EmbedBatch | None, Exception | None]] = []
        for chunk in chunks:
            try:
                batch = embedder.embed(chunk, model=job.embedding.model, dimensions=request_dims)
                results.append((batch, None))
            except Exception as exc:
                results.append((None, exc))
        return results

    ordered: list[tuple[EmbedBatch | None, Exception | None]] = [(None, None)] * len(chunks)

    def _run(offset: int, chunk: list[str]) -> tuple[int, EmbedBatch | None, Exception | None]:
        try:
            batch = embedder.embed(chunk, model=job.embedding.model, dimensions=request_dims)
            return offset, batch, None
        except Exception as exc:
            return offset, None, exc

    with ThreadPoolExecutor(max_workers=job.embedding.concurrency) as pool:
        futures = [pool.submit(_run, offset, chunk) for offset, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            offset, batch, error = future.result()
            ordered[offset] = (batch, error)
    return ordered


def _chunk(items: list[tuple[str, str]], size: int) -> list[list[tuple[str, str]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _group_windows(
    chunks: list[list[tuple[str, str]]], concurrency: int
) -> list[list[list[tuple[str, str]]]]:
    width = max(1, concurrency)
    return [chunks[index : index + width] for index in range(0, len(chunks), width)]


def _materialize_dataset(
    job: Job,
    loaded: LoadedDataset,
    store: JobStore,
    records: dict[tuple[str, int], EmbeddingRecord],
) -> Path:
    parts = DatasetDict()
    for split in loaded.splits:
        table: dict[str, list[object]] = {name: [] for name in split.columns}
        table[job.embedding.column] = []
        for index, row in enumerate(split.rows):
            for name in split.columns:
                table[name].append(row.get(name))
            record = records.get((split.name, index))
            table[job.embedding.column].append(record.embedding if record else None)
        parts[split.name] = Dataset.from_dict(table)
    dataset = parts
    output = store.output_dir(job.id)
    if output.exists():
        for child in sorted(output.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    dataset.save_to_disk(str(output))
    return output


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _key(job: Job, text: str) -> str:
    return cache_key(
        job.embedding.provider,
        job.embedding.model,
        job.embedding.dimensions,
        text,
    )


def _progress_line(progress: JobProgress, elapsed: float) -> str:
    return (
        f"embedded={progress.embedded} skipped={progress.skipped} "
        f"failed={progress.failed} tokens={progress.tokens} "
        f"cost=${progress.cost_usd:.6f} duration={elapsed:.1f}s"
    )


def _plan_with_batch(plan: Plan, batch_size: int, concurrency: int) -> Plan:
    embedding = plan.embedding
    updated = type(embedding)(
        provider=embedding.provider,
        model=embedding.model,
        dimensions=embedding.dimensions,
        column=embedding.column,
        source_columns=embedding.source_columns,
        batch_size=batch_size,
        concurrency=concurrency,
    )
    return Plan(
        source=plan.source,
        embedding=updated,
        output=plan.output,
        estimates=plan.estimates,
        created_at=plan.created_at,
    )
