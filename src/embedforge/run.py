"""Run and resume embedding jobs, writing local artifacts."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

from embedforge import __version__
from embedforge.cache import EmbeddingCache, cache_key
from embedforge.catalog import estimate_cost_usd, resolve_model
from embedforge.config import apply_overrides, load_config
from embedforge.errors import EmbedForgeError
from embedforge.hfdata import DatasetRequest, DatasetSource, HuggingFaceDatasetSource
from embedforge.plan import build_plan
from embedforge.providers import EmbedBatch, Embedder, get_embedder
from embedforge.shapes import (
    Config,
    Job,
    JobProgress,
    JobStatus,
    Plan,
    Provenance,
    replace_job,
    utc_now,
)
from embedforge.store import JobStore, new_ulid

ProgressCallback = Callable[[str], None]


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
        plan = job_store.load_plan(resume_id)
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
    column = job.embedding.source_columns[0]
    texts = [_as_text(row.get(column)) for row in loaded.rows]
    existing = store.load_embeddings(job.id)
    progress = job.progress
    started = time.monotonic()
    job = replace_job(job, status=JobStatus.RUNNING, error=None)
    store.save_job(job)

    try:
        next_index = progress.next_index
        while next_index < len(texts):
            window = min(
                job.embedding.batch_size * job.embedding.concurrency,
                len(texts) - next_index,
            )
            indexes = list(range(next_index, next_index + window))
            progress = _embed_window(
                texts=texts,
                indexes=indexes,
                job=job,
                store=store,
                cache=cache,
                embedder=embedder,
                existing=existing,
                progress=progress,
            )
            next_index += window
            job = replace_job(job, status=JobStatus.RUNNING, progress=progress)
            store.save_job(job)
            if on_progress:
                elapsed = time.monotonic() - started
                on_progress(_progress_line(progress, elapsed))
        output_path = _materialize_dataset(job, loaded.rows, loaded.columns, store)
        status = JobStatus.COMPLETED if progress.failed == 0 else JobStatus.FAILED
        job = replace_job(
            job,
            status=status,
            progress=progress,
            error=None if progress.failed == 0 else f"{progress.failed} rows failed",
        )
        store.save_job(job)
        if on_progress:
            elapsed = time.monotonic() - started
            on_progress(_progress_line(progress, elapsed))
            on_progress(f"output: {output_path}")
        return RunResult(job=job, output_path=str(output_path))
    except Exception as exc:
        job = replace_job(job, status=JobStatus.FAILED, error=str(exc))
        store.save_job(job)
        if isinstance(exc, EmbedForgeError):
            raise
        raise EmbedForgeError(str(exc)) from exc


def _embed_window(
    *,
    texts: list[str],
    indexes: list[int],
    job: Job,
    store: JobStore,
    cache: EmbeddingCache,
    embedder: Embedder,
    existing: dict[int, list[float] | None],
    progress: JobProgress,
) -> JobProgress:
    embedded = progress.embedded
    skipped = progress.skipped
    failed = progress.failed
    tokens = progress.tokens
    cost = progress.cost_usd
    pending: list[tuple[int, str]] = []
    resolved: dict[int, list[float] | None] = {}

    for index in indexes:
        if index in existing:
            skipped += 1
            resolved[index] = existing[index]
            continue
        text = texts[index]
        if text == "":
            skipped += 1
            resolved[index] = None
            store.append_embedding(job.id, index, None, None)
            existing[index] = None
            continue
        cached = cache.get(
            job.embedding.provider,
            job.embedding.model,
            job.embedding.dimensions,
            text,
        )
        if cached is not None:
            skipped += 1
            resolved[index] = cached
            key = _key(job, text)
            store.append_embedding(job.id, index, key, cached)
            existing[index] = cached
            continue
        pending.append((index, text))

    if pending:
        chunks = _chunk(pending, job.embedding.batch_size)
        results = _embed_chunks(chunks, job, embedder)
        model_info = resolve_model(
            job.embedding.provider, job.embedding.model, job.embedding.dimensions
        )
        for chunk, batch, error in results:
            if error is not None or batch is None:
                for index, text in chunk:
                    try:
                        single = embedder.embed(
                            [text],
                            model=job.embedding.model,
                            dimensions=job.embedding.dimensions,
                        )
                    except Exception:
                        failed += 1
                        resolved[index] = None
                        store.append_embedding(job.id, index, None, None)
                        existing[index] = None
                        continue
                    vector = single.vectors[0]
                    tokens += single.tokens
                    cost += estimate_cost_usd(single.tokens, model_info.usd_per_million_tokens)
                    _store_success(job, cache, store, existing, resolved, index, text, vector)
                    embedded += 1
                continue
            tokens += batch.tokens
            cost += estimate_cost_usd(batch.tokens, model_info.usd_per_million_tokens)
            for (index, text), vector in zip(chunk, batch.vectors, strict=True):
                _store_success(job, cache, store, existing, resolved, index, text, vector)
                embedded += 1

    return JobProgress(
        embedded=embedded,
        skipped=skipped,
        failed=failed,
        tokens=tokens,
        cost_usd=cost,
        next_index=indexes[-1] + 1 if indexes else progress.next_index,
    )


def _store_success(
    job: Job,
    cache: EmbeddingCache,
    store: JobStore,
    existing: dict[int, list[float] | None],
    resolved: dict[int, list[float] | None],
    index: int,
    text: str,
    vector: list[float],
) -> None:
    key = cache.put(
        job.embedding.provider,
        job.embedding.model,
        job.embedding.dimensions,
        text,
        vector,
    )
    store.append_embedding(job.id, index, key, vector)
    existing[index] = vector
    resolved[index] = vector


def _embed_chunks(
    chunks: list[list[tuple[int, str]]],
    job: Job,
    embedder: Embedder,
) -> list[tuple[list[tuple[int, str]], EmbedBatch | None, Exception | None]]:
    if job.embedding.concurrency <= 1 or len(chunks) == 1:
        results: list[tuple[list[tuple[int, str]], EmbedBatch | None, Exception | None]] = []
        for chunk in chunks:
            try:
                batch = embedder.embed(
                    [text for _, text in chunk],
                    model=job.embedding.model,
                    dimensions=job.embedding.dimensions,
                )
                results.append((chunk, batch, None))
            except Exception as exc:
                results.append((chunk, None, exc))
        return results

    results_by_order: list[tuple[list[tuple[int, str]], EmbedBatch | None, Exception | None]] = [
        (chunk, None, None) for chunk in chunks
    ]

    def _run(
        offset: int, chunk: list[tuple[int, str]]
    ) -> tuple[int, EmbedBatch | None, Exception | None]:
        try:
            batch = embedder.embed(
                [text for _, text in chunk],
                model=job.embedding.model,
                dimensions=job.embedding.dimensions,
            )
            return offset, batch, None
        except Exception as exc:
            return offset, None, exc

    with ThreadPoolExecutor(max_workers=job.embedding.concurrency) as pool:
        futures = [pool.submit(_run, offset, chunk) for offset, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            offset, batch, error = future.result()
            results_by_order[offset] = (chunks[offset], batch, error)
    return results_by_order


def _chunk(items: list[tuple[int, str]], size: int) -> list[list[tuple[int, str]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _materialize_dataset(
    job: Job,
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
    store: JobStore,
) -> Path:
    embeddings = store.load_embeddings(job.id)
    table: dict[str, list[object]] = {name: [] for name in columns}
    table[job.embedding.column] = []
    for index, row in enumerate(rows):
        for name in columns:
            table[name].append(row.get(name))
        table[job.embedding.column].append(embeddings.get(index))
    dataset = Dataset.from_dict(table)
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
